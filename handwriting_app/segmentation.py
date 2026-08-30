"""Split captured ink into words using the pen-lift structure.

Each pen-down..pen-up trace is a :class:`Stroke`. Words are groups of strokes
separated by a horizontal gap that is large relative to the writing height.
Recognizing one word at a time is much more reliable than one big line image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from handwriting_app.ink import Ink, Stroke


@dataclass
class _Box:
    index: int
    stroke: Stroke
    x0: float
    x1: float


def _stroke_box(index: int, stroke: Stroke) -> _Box:
    xs = [p[0] for p in stroke.points]
    return _Box(index=index, stroke=stroke, x0=min(xs), x1=max(xs))


def segment_words(
    ink: Ink,
    *,
    gap_ratio: float = 0.4,
    min_gap_px: float = 12.0,
) -> List[Ink]:
    """Return one :class:`Ink` per detected word, left to right.

    ``gap_ratio`` is the inter-word gap threshold as a fraction of the overall
    ink height; ``min_gap_px`` is a floor so tiny writing still segments sanely.
    Returns a single-element list (the whole ink) when nothing splits.
    """
    boxes = [
        _stroke_box(i, s) for i, s in enumerate(ink.strokes) if len(s) > 0
    ]
    if not boxes:
        return []

    bounds = ink.bounds()
    assert bounds is not None  # guaranteed by the non-empty boxes above
    line_height = max(1.0, bounds[3] - bounds[1])
    threshold = max(min_gap_px, gap_ratio * line_height)

    boxes.sort(key=lambda b: b.x0)

    groups: List[List[_Box]] = [[boxes[0]]]
    running_right = boxes[0].x1
    for box in boxes[1:]:
        if box.x0 - running_right > threshold:
            groups.append([box])
        else:
            groups[-1].append(box)
        running_right = max(running_right, box.x1)

    words: List[Ink] = []
    for group in groups:
        word = Ink()
        for box in sorted(group, key=lambda b: b.index):
            word.strokes.append(box.stroke)
        words.append(word)
    return words
