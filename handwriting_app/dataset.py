"""Read and write handwriting samples collected in training mode.

A sample is one JSON file under ``data/samples/`` holding the label and the raw
strokes, plus a PNG preview for eyeballing. Strokes are the source of truth so
that changes to rendering / deslanting flow through to training automatically.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Union

from handwriting_app import __version__
from handwriting_app.ink import Ink

PathLike = Union[str, Path]
_UNSAFE = re.compile(r"[^a-zA-Z0-9]+")
MANIFEST = "manifest.jsonl"


@dataclass
class Sample:
    label: str
    ink: Ink
    stroke_width: int = 8


def _slug(label: str, max_len: int = 24) -> str:
    slug = _UNSAFE.sub("-", label).strip("-").lower()
    return slug[:max_len] or "sample"


def count_samples(samples_dir: PathLike) -> int:
    directory = Path(samples_dir)
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob("*.json"))


def save_sample(
    ink: Ink,
    label: str,
    samples_dir: PathLike,
    *,
    stroke_width: int = 8,
) -> Path:
    directory = Path(samples_dir)
    directory.mkdir(parents=True, exist_ok=True)

    index = count_samples(directory) + 1
    stem = f"{index:04d}_{_slug(label)}"
    path = directory / f"{stem}.json"

    record = {
        "label": label,
        "strokes": ink.to_dict()["strokes"],
        "stroke_width": stroke_width,
        "app_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    try:  # preview image is best-effort
        preview = ink.render(stroke_width=stroke_width, deslant=False)
        if preview is not None:
            preview.save(directory / f"{stem}.png")
    except Exception:  # noqa: BLE001
        pass

    with (directory / MANIFEST).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"file": path.name, "label": label}) + "\n")
    return path


def load_sample(path: PathLike) -> Sample:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Sample(
        label=data["label"],
        ink=Ink.from_dict(data),
        stroke_width=int(data.get("stroke_width", 8)),
    )


def iter_samples(samples_dir: PathLike) -> Iterator[Sample]:
    for path in sorted(Path(samples_dir).glob("*.json")):
        yield load_sample(path)
