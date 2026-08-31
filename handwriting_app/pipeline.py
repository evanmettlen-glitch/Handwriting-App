"""Ink -> text: segment into words, recognize each, correct against a lexicon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from handwriting_app.calibration import Calibration
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
    deslant: bool = True
    spellcheck: bool = True
    spell_compound: bool = False
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
        # Calibration overrides the render settings it was measured with.
        self.segment = config.segment
        cal = config.calibration
        self._deslant = cal.deslant if cal else config.deslant
        self._stroke_width = cal.stroke_width if cal else config.stroke_width
        self._render_pad = cal.render_pad if cal else config.render_pad
        self._smooth = cal.smooth if cal else config.smooth
        self._word_gap_ratio = cal.word_gap_ratio if cal else config.word_gap_ratio

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
        return notes

    def run(self, ink: Ink) -> str:
        if ink.is_empty:
            return ""

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
            image = word.render(
                stroke_width=self._stroke_width,
                pad=self._render_pad,
                deslant=self._deslant,
                smooth=self._smooth,
            )
            if image is None:
                continue
            text = self.recognizer.recognize(image, hint=hint).strip()
            if text:
                pieces.append(text)

        line = " ".join(pieces)
        if self.config.calibration is not None:
            line = self.config.calibration.apply_fixes(line)
        if self._corrector is not None and self._corrector.available:
            line = self._corrector.correct_line(
                line, compound=self.config.spell_compound
            )
        return line
