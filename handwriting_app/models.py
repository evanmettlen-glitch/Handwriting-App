"""Locate which recognition model the app should load.

Priority:
  1. an explicit --model-dir
  2. models/<user>-onnx        (a model fine-tuned for --user)
  3. models/trocr-personal-onnx
  4. models/trocr-base-handwritten-onnx
  5. models/trocr-small-handwritten-onnx   (also the fallback path if none exist)
"""

from __future__ import annotations

from pathlib import Path

from handwriting_app.naming import user_slug

GENERIC_CANDIDATES = (
    "models/personal-onnx",
    "models/trocr-personal-onnx",
    "models/trocr-base-handwritten-onnx",
    "models/trocr-small-handwritten-onnx",
)


def resolve_model_dir(explicit: str = "", user: str = "") -> str:
    if explicit:
        return explicit
    if user:
        personal = f"models/{user_slug(user)}-onnx"
        if Path(personal).is_dir():
            return personal
    for candidate in GENERIC_CANDIDATES:
        if Path(candidate).is_dir():
            return candidate
    return GENERIC_CANDIDATES[-1]


def model_exists(explicit: str = "", user: str = "") -> bool:
    return Path(resolve_model_dir(explicit, user)).is_dir()
