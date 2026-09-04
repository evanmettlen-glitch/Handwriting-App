"""A tiny personalization layer that needs no model training.

Apple-style personalization: keep one strong general recognizer and adapt the
*inputs and outputs* around it, rather than retraining weights per user.

``calibration.json`` holds:
  render  - the render settings that scored best on this user's own samples
  fixes   - whole-word substitutions the recognizer reliably gets wrong for them
  words   - their vocabulary (also mined by lexicon.py)

Produced by ``scripts/calibrate.py`` in a single forward pass over the collected
samples — minutes, not epochs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

FILENAME = "calibration.json"


@dataclass
class Calibration:
    deslant: bool = True
    stroke_width: int = 8
    render_pad: int = 32
    smooth: bool = True
    word_gap_ratio: float = 0.4
    fixes: Dict[str, str] = field(default_factory=dict)
    baseline_cer: Optional[float] = None
    tuned_cer: Optional[float] = None
    samples: int = 0

    def to_dict(self) -> dict:
        return {
            "render": {
                "deslant": self.deslant,
                "stroke_width": self.stroke_width,
                "render_pad": self.render_pad,
                "smooth": self.smooth,
            },
            "word_gap_ratio": self.word_gap_ratio,
            "fixes": self.fixes,
            "baseline_cer": self.baseline_cer,
            "tuned_cer": self.tuned_cer,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        render = data.get("render", {})
        return cls(
            deslant=bool(render.get("deslant", True)),
            stroke_width=int(render.get("stroke_width", 8)),
            render_pad=int(render.get("render_pad", 32)),
            smooth=bool(render.get("smooth", True)),
            word_gap_ratio=float(data.get("word_gap_ratio", 0.4)),
            fixes={str(k): str(v) for k, v in (data.get("fixes") or {}).items()},
            baseline_cer=data.get("baseline_cer"),
            tuned_cer=data.get("tuned_cer"),
            samples=int(data.get("samples", 0)),
        )

    def apply_fixes(self, text: str) -> str:
        if not self.fixes or not text:
            return text
        out = []
        for token in text.split(" "):
            replacement = self.fixes.get(token) or self.fixes.get(token.lower())
            out.append(replacement if replacement else token)
        return " ".join(out)


def path_for(samples_dir: str) -> Path:
    return Path(samples_dir) / FILENAME


def save(calibration: Calibration, samples_dir: str) -> Path:
    target = path_for(samples_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return target


def load(samples_dir: str) -> Optional[Calibration]:
    target = path_for(samples_dir)
    if not target.is_file():
        return None
    try:
        return Calibration.from_dict(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, AttributeError):
        # Wrong-shape JSON (a list, a string) parses but then fails on attribute
        # or item access. A hand-edited calibration.json must degrade to "no
        # calibration", never take the recognizer down with it.
        return None
