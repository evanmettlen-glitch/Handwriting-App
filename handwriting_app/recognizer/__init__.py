"""Recognition backends: image of ink in, text out."""

from __future__ import annotations

import os

from handwriting_app.config import AppConfig

from .base import RecognitionError, Recognizer

__all__ = ["RecognitionError", "Recognizer", "build_recognizer", "resolve_backend"]


def resolve_backend(config: AppConfig) -> str:
    """Turn ``backend="auto"`` into a concrete choice.

    Prefer TrOCR when its exported model is present and its deps import;
    otherwise fall back to tesseract.
    """
    if config.backend != "auto":
        return config.backend
    if os.path.isdir(config.model_dir):
        try:
            import optimum.onnxruntime  # noqa: F401
            import transformers  # noqa: F401

            return "trocr"
        except ImportError:
            pass
    return "tesseract"


def build_recognizer(config: AppConfig) -> Recognizer:
    backend = resolve_backend(config)
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
