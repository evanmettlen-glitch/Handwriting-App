"""Streaming partial text out of the decoder, and guessing where it is going."""

import pytest

from handwriting_app.ink import Ink, Stroke
from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline
from handwriting_app.postprocess import SpellCorrector
from handwriting_app.recognizer.base import Recognizer


def _ink(*x_starts):
    """One tall stroke per word, far enough apart to segment."""
    ink = Ink()
    for x0 in x_starts or (0,):
        ink.strokes.append(Stroke([(x0, 0.0), (x0 + 20.0, 40.0), (x0 + 40.0, 0.0)]))
    return ink


class Talker(Recognizer):
    """A recognizer that streams its answer one character at a time."""

    name = "talker"

    def __init__(self, *words: str) -> None:
        self.words = list(words) or ["hello"]
        self.calls = 0

    def recognize(self, image, *, hint="line", on_partial=None):
        word = self.words[min(self.calls, len(self.words) - 1)]
        self.calls += 1
        if on_partial is not None:
            for size in range(1, len(word) + 1):
                on_partial(word[:size])
        return word


class Mute(Recognizer):
    """A recognizer written before streaming existed — no ``on_partial``."""

    name = "mute"

    def recognize(self, image, *, hint="line"):
        return "hi"


def _pipeline(recognizer, **kw):
    return RecognitionPipeline(recognizer, PipelineConfig(spellcheck=False, **kw))


# -- streaming ------------------------------------------------------------


def test_partials_reach_the_caller_as_the_decoder_produces_them():
    seen = []
    pipe = _pipeline(Talker("hello"), segment=False)
    assert pipe.run(_ink(), on_partial=seen.append) == "hello"
    assert seen == ["h", "he", "hel", "hell", "hello"]


def test_word_mode_partials_carry_the_whole_line_so_far():
    """Mid-decode the preview should read 'hello wor', not 'wor'."""
    seen = []
    pipe = _pipeline(Talker("hello", "world"), segment=True)
    pipe.run(_ink(0, 400), on_partial=seen.append)
    assert seen[:2] == ["h", "he"]
    assert seen[-2:] == ["hello worl", "hello world"]


def test_a_recognizer_that_cannot_stream_is_never_asked_to():
    """The argument is only passed when a caller wants partials, so backends
    predating it keep working."""
    assert _pipeline(Mute(), segment=False).run(_ink()) == "hi"
    with pytest.raises(TypeError):
        _pipeline(Mute(), segment=False).run(_ink(), on_partial=lambda _t: None)


def test_running_without_a_callback_streams_nothing():
    talker = Talker("hello")
    assert _pipeline(talker, segment=False).run(_ink()) == "hello"
    assert talker.calls == 1


# -- word completion ------------------------------------------------------

_HAS_DICT = SpellCorrector().available
needs_dict = pytest.mark.skipif(_HAS_DICT is False, reason="symspellpy not installed")


@needs_dict
def test_complete_extends_a_fragment_to_a_real_word():
    corrector = SpellCorrector()
    guess = corrector.complete("hel")
    assert guess.startswith("hel") and len(guess) > 3


@needs_dict
def test_complete_prefers_your_own_vocabulary():
    """The English frequency counts run to billions, so a personal word has to
    win outright rather than on weight."""
    plain = SpellCorrector()
    personal = SpellCorrector(boost={"priya": 9})
    assert personal.complete("pri") == "priya"
    assert plain.complete("pri") != "priya"


@needs_dict
def test_complete_matches_the_case_of_the_fragment():
    assert SpellCorrector().complete("Hel").istitle()


@needs_dict
def test_the_prefix_index_agrees_with_scanning_the_whole_dictionary():
    """complete() buckets the dictionary by first two letters so it can run
    once per generated token. The shortcut has to give the same answer."""
    corrector = SpellCorrector()
    words = corrector._sym.words  # noqa: SLF001 - the point is the shortcut

    def brute(prefix: str) -> str:
        lower = prefix.lower()
        best, best_count = "", -1
        for word, count in words.items():
            if count > best_count and len(word) > len(lower) and word.startswith(lower):
                best, best_count = word, count
        return best

    for prefix in ("th", "brow", "hel", "qui", "xyl", "zz", "understan"):
        assert corrector.complete(prefix) == brute(prefix), prefix


@needs_dict
def test_complete_declines_when_there_is_nothing_to_go_on():
    corrector = SpellCorrector()
    assert corrector.complete("h") == ""  # one letter is not a prefix
    assert corrector.complete("zzq") == ""  # no word starts like that
    assert corrector.complete("h3") == ""  # not a word fragment at all


# -- prediction policy ----------------------------------------------------


@needs_dict
def test_predict_returns_only_the_guessed_tail():
    pipe = RecognitionPipeline(Talker(), PipelineConfig())
    assert pipe.predict("the qui") == "ck"


@needs_dict
def test_predict_does_not_guess_past_a_finished_word():
    """'the' is a word. Turning it into 'the(y)' is noise, not help."""
    pipe = RecognitionPipeline(Talker(), PipelineConfig())
    assert pipe.predict("the") == ""
    assert pipe.predict("the quick ") == ""


@needs_dict
def test_predict_is_off_when_disabled_or_without_a_dictionary():
    assert RecognitionPipeline(Talker(), PipelineConfig(predict=False)).predict("qui") == ""
    assert RecognitionPipeline(
        Talker(), PipelineConfig(spellcheck=False)
    ).predict("qui") == ""


def test_predict_never_changes_the_committed_text():
    """Whatever the guess is, run() returns what the model actually produced."""
    pipe = RecognitionPipeline(Talker("hel"), PipelineConfig(segment=False))
    assert pipe.run(_ink()) == "hel"


# -- flags ----------------------------------------------------------------


def test_preview_flags_parse():
    from handwriting_app.config import parse_args

    assert parse_args([]).predict is True
    assert parse_args(["--no-predict"]).predict is False
