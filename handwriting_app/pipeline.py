"""Ink -> text: segment into words, recognize each, correct against a lexicon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from handwriting_app.ink import Ink
from handwriting_app.postprocess import SpellCorrector
from handwriting_app.recognizer import Recognizer
from handwriting_app.segmentation import segment_words


@dataclass
class PipelineConfig:
    segment: bool = True
    word_gap_ratio: float = 0.4
    deslant: bool = True
    spellcheck: bool = True
    spell_compound: bool = False
    stroke_width: int = 8
    render_pad: int = 32


class RecognitionPipeline:
    def __init__(self, recognizer: Recognizer, config: PipelineConfig) -> None:
        self.recognizer = recognizer
        self.config = config
        self._corrector = SpellCorrector() if config.spellcheck else None

    @property
    def notes(self) -> List[str]:
        notes: List[str] = []
        if self.config.spellcheck and (
            self._corrector is None or not self._corrector.available
        ):
            notes.append("dictionary correction off (pip install symspellpy)")
        return notes

    def run(self, ink: Ink) -> str:
        if ink.is_empty:
            return ""

        words = (
            segment_words(ink, gap_ratio=self.config.word_gap_ratio)
            if self.config.segment
            else [ink]
        )
        if not words:
            words = [ink]
        hint = "word" if len(words) > 1 or self.config.segment else "line"

        pieces: List[str] = []
        for word in words:
            image = word.render(
                stroke_width=self.config.stroke_width,
                pad=self.config.render_pad,
                deslant=self.config.deslant,
            )
            if image is None:
                continue
            text = self.recognizer.recognize(image, hint=hint).strip()
            if text:
                pieces.append(text)

        line = " ".join(pieces)
        if self._corrector is not None and self._corrector.available:
            line = self._corrector.correct_line(
                line, compound=self.config.spell_compound
            )
        return line
