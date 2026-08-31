"""Recognition backends: image of ink in, text out."""

from __future__ import annotations

import importlib.util

from handwriting_app.config import AppConfig
from handwriting_app.models import is_onnx_dir, resolve_model_dir

from .base import RecognitionError, Recognizer

__all__ = [
    "RecognitionError",
    "Recognizer",
    "build_recognizer",
    "resolve_backend",
]


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


def resolve_backend(config: AppConfig) -> str:
    """Turn ``backend="auto"`` into a concrete choice.

    TrOCR needs only torch + transformers (the ONNX path is an optimization),
    so prefer it whenever those are installed; otherwise fall back to tesseract.
    """
    if config.backend != "auto":
        return config.backend
    if _installed("torch", "transformers"):
        return "trocr"
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

    if backend in ("trocr", "trocr-onnx", "trocr-torch"):
        model_ref = resolve_model_dir(config.model_dir, config.user)

        # Use the ONNX runtime only for a directory that actually holds an
        # exported model AND has optimum available; otherwise load with torch,
        # which needs no export step.
        use_onnx = (
            backend != "trocr-torch"
            and is_onnx_dir(model_ref)
            and _installed("optimum")
        )
        if use_onnx:
            from .trocr_onnx_recognizer import TrocrOnnxRecognizer

            return TrocrOnnxRecognizer(model_dir=model_ref)

        if backend == "trocr-onnx":
            raise RecognitionError(
                f"'{model_ref}' is not an exported ONNX model, or optimum is "
                "not installed. Use --backend trocr to load it with torch."
            )

        from .trocr_torch_recognizer import TrocrTorchRecognizer

        return TrocrTorchRecognizer(model_dir=model_ref)

    raise RecognitionError(f"Unknown backend: {backend!r}")
