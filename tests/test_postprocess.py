from handwriting_app.postprocess import SpellCorrector, _match_case


def test_match_case():
    assert _match_case("HELLO", "world") == "WORLD"
    assert _match_case("Hello", "world") == "World"
    assert _match_case("hello", "world") == "world"


def test_corrector_is_a_noop_when_unavailable():
    corrector = SpellCorrector()
    if corrector.available:
        return  # covered by the dictionary test below
    text = "teh quikc bronw fox"
    assert corrector.correct_line(text) == text


def test_corrector_fixes_typos_when_dictionary_present():
    corrector = SpellCorrector()
    if not corrector.available:
        return
    assert corrector.correct_word("recieve") == "receive"
    assert corrector.correct_line("teh cat") == "the cat"
    # unknown-but-plausible tokens and casing are preserved
    assert corrector.correct_word("Xyzzy").istitle()


def test_join_split_letters():
    """Widely-spaced printing reads back as one-letter words."""
    c = SpellCorrector()
    if not c.available:
        return
    assert c.join_split_letters("a n d") == "and"
    assert c.join_split_letters("w i t h") == "with"
    assert c.join_split_letters("the quick b r o w n") == "the quick brown"
    # a run may hold several words
    assert c.join_split_letters("I a m h e r e") == "I am here"


def test_join_split_letters_leaves_valid_text_alone():
    c = SpellCorrector()
    if not c.available:
        return
    for text in (
        "the",
        "you",
        "a",
        "I will call you later",
        "fox , in the",
        "the quick-",
    ):
        assert c.join_split_letters(text) == text


def test_join_split_letters_keeps_unjoinable_runs():
    c = SpellCorrector()
    if not c.available:
        return
    # nothing in the run forms a word, so don't invent one
    assert c.join_split_letters("x q z") == "x q z"


def test_join_runs_before_correction_in_the_pipeline():
    from handwriting_app.ink import Ink
    from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline
    from handwriting_app.recognizer.base import Recognizer

    class Fake(Recognizer):
        name = "fake"

        def recognize(self, image, *, hint="line"):
            return "a n d"

    ink = Ink()
    stroke = ink.start_stroke()
    for x in range(0, 40, 4):
        stroke.add(x, 20)

    joined = RecognitionPipeline(Fake(), PipelineConfig(segment=False))
    if joined._corrector is None or not joined._corrector.available:
        return
    assert joined.run(ink) == "and"

    off = RecognitionPipeline(
        Fake(), PipelineConfig(segment=False, join_letters=False)
    )
    assert off.run(ink) != "and"


def test_personal_lexicon_protects_own_words_but_still_fixes_typos():
    plain = SpellCorrector()
    if not plain.available:
        return
    boosted = SpellCorrector(boost={"priya": 3, "homehub": 1})
    assert boosted.boosted == 2
    # a real name the generic dictionary would otherwise "correct"
    assert plain.correct_word("Priya") != "Priya"
    assert boosted.correct_word("Priya") == "Priya"
    assert boosted.correct_word("homehub") == "homehub"
    # genuine typos are still fixed
    assert boosted.correct_word("teh") == "the"
