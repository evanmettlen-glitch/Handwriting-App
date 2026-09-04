"""Ink -> text: segment into words, recognize each, correct against a lexicon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from handwriting_app.calibration import Calibration
from handwriting_app.cleanup import CleanupConfig, CleanupReport, clean_ink
from handwriting_app.ink import Ink
from handwriting_app.postprocess import SpellCorrector
from handwriting_app.recognizer import Recognizer
from handwriting_app.segmentation import segment_words


def resolve_segment(setting: Optional[bool], recognizer_name: str) -> bool:
    """Whether to split ink into words before recognition.

    ``None`` means auto. TrOCR was trained on IAM *lines* and its decoder uses
    cross-word context, so feeding it whole lines beats feeding it words —
    segmentation only helps tesseract, which is weak on multi-word images.
    """
    if setting is not None:
        return setting
    return not recognizer_name.startswith("trocr")


@dataclass
class PipelineConfig:
    segment: bool = True
    word_gap_ratio: float = 0.4
    cleanup: bool = True
    cleanup_config: Optional[CleanupConfig] = None
    predict: bool = True
    deslant: bool = True
    spellcheck: bool = True
    spell_compound: bool = False
    join_letters: bool = True
    stroke_width: int = 8
    render_pad: int = 32
    smooth: bool = True
    personal_lexicon: Dict[str, int] = field(default_factory=dict)
    calibration: Optional[Calibration] = None


class RecognitionPipeline:
    def __init__(self, recognizer: Recognizer, config: PipelineConfig) -> None:
        self.recognizer = recognizer
        self.config = config
        self._corrector = (
            SpellCorrector(boost=config.personal_lexicon or None)
            if config.spellcheck
            else None
        )
        # Calibration overrides these render settings UNCONDITIONALLY when
        # present — including a value the caller passed explicitly, e.g.
        # `--stroke-width 12` after `scripts/calibrate.py` has written
        # calibration.json silently loses to whatever calibration measured.
        # There is no "the user asked for this on purpose" signal at this
        # layer to check first: AppConfig's fields carry only the resolved
        # value, not whether argparse's default was overridden, so distinguishing
        # them would mean threading that through config.py and app.py as well —
        # out of scope for this constructor. `--no-calibration` is the escape
        # hatch and the *only* one; it is called out here because that is easy
        # to miss from the CLI help text alone.
        self.segment = config.segment
        cal = config.calibration
        self._deslant = cal.deslant if cal else config.deslant
        self._stroke_width = cal.stroke_width if cal else config.stroke_width
        self._render_pad = cal.render_pad if cal else config.render_pad
        self._smooth = cal.smooth if cal else config.smooth
        self._word_gap_ratio = cal.word_gap_ratio if cal else config.word_gap_ratio
        self._cleanup = config.cleanup_config or CleanupConfig()
        #: Report from the most recent :meth:`prepare`, for the status line.
        self.last_cleanup: Optional[CleanupReport] = None

    @property
    def notes(self) -> List[str]:
        notes: List[str] = []
        if self.config.spellcheck and (
            self._corrector is None or not self._corrector.available
        ):
            notes.append("dictionary correction off (pip install symspellpy)")
        elif self._corrector is not None and self._corrector.boosted:
            notes.append(f"personal lexicon: {self._corrector.boosted} words")
        cal = self.config.calibration
        if cal is not None:
            note = f"calibrated on {cal.samples} samples"
            if cal.tuned_cer is not None:
                note += f" (CER {cal.tuned_cer:.2f})"
            notes.append(note)
        notes.append("word-by-word" if self.segment else "whole line")
        if not self.config.cleanup:
            notes.append("ink cleanup off")
        return notes

    def prepare(self, ink: Ink) -> Ink:
        """Drop drags, no-lift connectors and stray marks before recognition.

        Kept separate from :meth:`run` so tooling can measure and visualize the
        cleaned ink — it is what the recognizer actually sees.
        """
        if not self.config.cleanup:
            self.last_cleanup = None
            return ink
        cleaned, report = clean_ink(ink, self._cleanup)
        self.last_cleanup = report
        return cleaned

    def render(self, ink: Ink):
        """Rasterize ink with the app's render settings, calibration included.

        Tooling (benchmarks, calibration, inspection) needs the same image the
        recognizer will actually see. Pass the ink through :meth:`prepare`
        first for that — :meth:`run` does, and cleanup changes the picture.
        """
        return ink.render(
            stroke_width=self._stroke_width,
            pad=self._render_pad,
            deslant=self._deslant,
            smooth=self._smooth,
        )

    def predict(self, partial: str) -> str:
        """Guess where a half-decoded line is going, for display only.

        The decoder emits a token at a time, so a preview mid-word reads as a
        fragment. Completing that fragment from the dictionary and the personal
        lexicon makes the preview legible sooner. It costs nothing on the
        recognition path and it changes nothing: the committed text is always
        what the model actually produced.

        Returns the completion of the trailing word, or ``""`` when there is no
        guess worth showing.
        """
        if not self.config.predict or self._corrector is None:
            return ""
        if not self._corrector.available or not partial or partial.endswith(" "):
            return ""
        tail = partial.rsplit(" ", 1)[-1]
        # A tail that is already a word is not a fragment. Guessing past it
        # turns every "the" into "the(y)", which is noise, not help.
        if self._corrector.knows(tail):
            return ""
        guess = self._corrector.complete(tail)
        return guess[len(tail) :] if guess else ""

    def postprocess(self, line: str) -> str:
        """Fixes, letter-joining, and dictionary correction applied to raw text."""
        if self.config.calibration is not None:
            line = self.config.calibration.apply_fixes(line)
        if self._corrector is not None and self._corrector.available:
            # Join before correcting: "a n d" must become "and" while the
            # letters are still adjacent tokens.
            if self.config.join_letters:
                line = self._corrector.join_split_letters(line)
            line = self._corrector.correct_line(
                line, compound=self.config.spell_compound
            )
        return line

    def run(
        self, ink: Ink, on_partial: Optional[Callable[[str], None]] = None
    ) -> str:
        if ink.is_empty:
            return ""

        ink = self.prepare(ink)
        words = (
            segment_words(ink, gap_ratio=self._word_gap_ratio)
            if self.segment
            else [ink]
        )
        if not words:
            words = [ink]
        hint = "word" if self.segment else "line"

        pieces: List[str] = []
        for word in words:
            image = self.render(word)
            if image is None:
                continue
            extra = {}
            if on_partial is not None:
                # Word-by-word: the caller wants the whole line so far, not the
                # fragment of whichever word is currently being decoded. Passed
                # only when asked for, so a recognizer that cannot stream never
                # has to know the argument exists.
                done = " ".join(pieces)
                extra["on_partial"] = (
                    lambda part, done=done: on_partial(f"{done} {part}".strip())
                )
            text = self.recognizer.recognize(image, hint=hint, **extra).strip()
            if text:
                pieces.append(text)

        return self.postprocess(" ".join(pieces))
