"""Locate which recognition model the app should load.

Priority:
  1. an explicit --model-dir (local path or a Hugging Face model id)
  2. models/<user>-onnx or models/<user>   (fine-tuned for --user)
  3. models/trocr-personal[-onnx], models/trocr-base-handwritten-onnx, ...
  4. the default Hugging Face id, downloaded and cached on first run

A returned value is either a local directory or a HF model id; use
:func:`is_local` / :func:`is_onnx_dir` to tell which loader to use.
"""

from __future__ import annotations

from pathlib import Path

from handwriting_app.naming import user_slug

# Used when nothing has been exported or fine-tuned locally. transformers
# downloads and caches it on first use (~1.3 GB for base, ~250 MB for small).
DEFAULT_HF_MODEL = "microsoft/trocr-base-handwritten"

GENERIC_CANDIDATES = (
    "models/personal-onnx",
    "models/personal",
    "models/trocr-personal-onnx",
    "models/trocr-personal",
    "models/trocr-base-handwritten-onnx",
    "models/trocr-small-handwritten-onnx",
)


def is_local(model_ref: str) -> bool:
    return Path(model_ref).is_dir()


def is_onnx_dir(model_ref: str) -> bool:
    """True when the directory holds an exported ONNX model."""
    path = Path(model_ref)
    return path.is_dir() and any(path.glob("encoder_model*.onnx"))


def resolve_model_dir(explicit: str = "", user: str = "") -> str:
    if explicit:
        return explicit
    if user:
        slug = user_slug(user)
        for candidate in (f"models/{slug}-onnx", f"models/{slug}"):
            if Path(candidate).is_dir():
                return candidate
    for candidate in GENERIC_CANDIDATES:
        if Path(candidate).is_dir():
            return candidate
    return DEFAULT_HF_MODEL
