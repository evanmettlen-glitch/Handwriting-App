"""Recognition backends: image of ink in, text out."""

from __future__ import annotations

from handwriting_app.config import AppConfig

from .base import RecognitionError, Recognizer

__all__ = ["RecognitionError", "Recognizer", "build_recognizer"]


def build_recognizer(config: AppConfig) -> Recognizer:
    backend = config.backend
    if backend == "tesseract":
        from .tesseract_recognizer import TesseractRecognizer

        return TesseractRecognizer(
            lang=config.lang,
            psm=config.psm,
            whitelist=config.whitelist or None,
        )
    if backend in ("trocr", "trocr-onnx"):
        from .trocr_onnx_recognizer import TrocrOnnxRecognizer

        return TrocrOnnxRecognizer(model_dir=config.model_dir)
    raise RecognitionError(f"Unknown backend: {backend!r}")
