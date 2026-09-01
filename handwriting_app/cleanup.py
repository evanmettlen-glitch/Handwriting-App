"""Strip non-writing marks out of captured ink before it reaches the recognizer.

Real touchscreen input is messier than the enrollment set suggests. Three
failure modes show up constantly and none of them are the recognizer's fault:

*Accidental drags* — a finger or sleeve brushing the pad leaves a long, flat,
almost perfectly straight line through the writing.

*No pen lift* — writing a whole phrase without lifting means the slide from the
end of one word to the start of the next is captured as ink. The recognizer
sees the words welded together, and :func:`segment_words` sees one stroke.

*Stray taps* — a knuckle or palm contact drops a speck off to the side, which
inflates the ink bounding box and shrinks the real writing when the image is
scaled to the model's input height.

All three are the same shape geometrically: a *traverse*, a long horizontal move
that carries no vertical information. Letters do not do that. Even a cursive
ligature between two letters is short (0.15-0.25x the line height) and rises and
falls; a traverse runs a line height or more and stays flat — or half that, if
it was also covered fast enough to have been a move rather than a mark. So one
detector handles drags and missing pen lifts alike — cut the traverse out and
the strokes on either side become separate words again.

Everything is measured as a fraction of the writing height, so the thresholds
hold for big and small handwriting alike. Nothing here mutates the input: split
strokes are new objects and untouched strokes are shared, matching what
:mod:`handwriting_app.segmentation` already does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from handwriting_app.ink import Ink, Point, Stroke


@dataclass(frozen=True)
class CleanupConfig:
    """Thresholds for :func:`clean_ink`, all relative to the writing height.

    The defaults are deliberately conservative: a false positive deletes the
    user's writing, which is far worse than leaving a drag in for the
    recognizer to cope with.
    """

    #: End-to-end length a traverse must reach to count as one.
    min_length: float = 1.0
    #: Length that suffices when the move was also fast (see ``fast_ratio``).
    fast_min_length: float = 0.5
    #: Steps this many times the typical writing step mean the pen was moving.
    fast_ratio: float = 3.0
    #: Total vertical extent a traverse is allowed over its whole length.
    max_rise: float = 0.15
    #: How one-way a traverse must be: |net dx| / total |dx| travelled.
    min_directness: float = 0.92
    #: Leftovers smaller than this after a cut are nubs, not letters.
    min_fragment: float = 0.08
    #: Bounding-box diagonal below which a stroke is a speck (an i-dot, or dirt).
    speck_size: float = 0.12
    #: Horizontal distance from all other ink that makes a speck a stray mark.
    speck_isolation: float = 0.80
    #: Vertical distance outside the writing band that makes a speck stray.
    speck_drift: float = 0.60


@dataclass(frozen=True)
class CleanupReport:
    """What :func:`clean_ink` changed, for the status line and diagnostics."""

    height: float = 0.0
    strokes_in: int = 0
    strokes_out: int = 0
    traverses_cut: int = 0
    traverse_pixels: float = 0.0
    specks_dropped: int = 0
    #: True when cleanup would have emptied the pad and was abandoned.
    reverted: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.traverses_cut or self.specks_dropped)

    def summary(self) -> str:
        """A short human-readable note, or ``""`` when nothing was touched."""
        if self.reverted:
            return "cleanup skipped (would have erased everything)"
        parts: List[str] = []
        if self.traverses_cut:
            noun = "drag" if self.traverses_cut == 1 else "drags"
            parts.append(f"{self.traverses_cut} {noun} cut")
        if self.specks_dropped:
            noun = "speck" if self.specks_dropped == 1 else "specks"
            parts.append(f"{self.specks_dropped} stray {noun} dropped")
        return ", ".join(parts)


def _percentile(ordered: Sequence[float], q: float) -> float:
    """``q``-quantile of an already-sorted sequence, linearly interpolated."""
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def writing_height(ink: Ink) -> float:
    """Height of the writing, ignoring outliers.

    The plain bounding box is the obvious measure and the wrong one: a single
    stray tap at the top of the pad doubles it, and every threshold here is a
    fraction of it. Trimming the outer 3% of points on each side keeps a stray
    mark from redefining what "tall" means.
    """
    ys = sorted(y for s in ink.strokes for _, y in s.points)
    if not ys:
        return 0.0
    return max(1.0, _percentile(ys, 0.97) - _percentile(ys, 0.03))


def _low_rise_end(points: Sequence[Point], start: int, limit: float) -> int:
    """Last index reachable from ``start`` without the run rising above ``limit``.

    Per-*step* flatness is the tempting test and it does not work: at real
    capture density a step along the leg of a letter is only a few pixels long
    and a couple of pixels of touch jitter make it look as flat as a drag. What
    separates the two is *cumulative* — over its whole length a traverse never
    leaves a narrow horizontal band, and no letter can say that.
    """
    low = high = points[start][1]
    end = start
    while end + 1 < len(points):
        y = points[end + 1][1]
        if max(high, y) - min(low, y) > limit:
            break
        low, high = min(low, y), max(high, y)
        end += 1
    return end


def _steps(points: Sequence[Point]) -> List[float]:
    return [math.dist(a, b) for a, b in zip(points, points[1:])]


def typical_step(ink: Ink) -> float:
    """Median distance between consecutive captured points while writing.

    Tk coalesces motion events, so how far apart the samples land is a proxy
    for how fast the pen was moving — measured at about 4.5 px for this user's
    normal writing. A slide between two words is several times that.
    """
    steps = sorted(step for s in ink.strokes for step in _steps(s.points) if step > 0)
    return _percentile(steps, 0.5) if steps else 0.0


def _is_traverse(
    span: Sequence[Point],
    height: float,
    step: float,
    cfg: CleanupConfig,
    *,
    interior: bool = False,
) -> bool:
    """Is this run of points a non-writing horizontal move?

    Length alone has to be set conservatively. A dash or a flourish can run
    flat for half a line height, and deleting one of those is worse than
    leaving a drag in. Two things together earn the shorter threshold:

    *interior* — there is real writing on both sides of it. A dash is its own
    stroke; a slide between two words is not, which is the whole problem.

    *fast* — somewhere in it the pen jumped several times further between
    samples than it does while writing. Tk's coalescing turns speed into sparse
    points, so a big stride is a move rather than a mark.

    Both at once is a pen that never lifted between two words. Neither is worth
    guessing about, so the run has to be a full line height instead.
    """
    chord = math.dist(span[0], span[-1])
    strides = _steps(span)
    fast = step > 0 and strides and max(strides) >= cfg.fast_ratio * step
    lenient = fast and interior
    if chord < (cfg.fast_min_length if lenient else cfg.min_length) * height:
        return False

    # Straightness in 2-D would be the obvious second test and it double-counts
    # the vertical wander the rise limit already caps — on a short traverse the
    # curl of the letters at either end then sinks it. What is left to check is
    # that the pen went one way: a scribbled-out word stays just as flat.
    travelled = sum(abs(b[0] - a[0]) for a, b in zip(span, span[1:]))
    return travelled > 0 and abs(span[-1][0] - span[0][0]) / travelled >= cfg.min_directness


def find_traverses(
    points: Sequence[Point],
    height: float,
    cfg: CleanupConfig,
    step: float = 0.0,
) -> List[Tuple[int, int]]:
    """Index ranges ``(i, j)`` of the traverses in one stroke, left to right.

    Both endpoints belong to the writing on either side, so a cut at ``(i, j)``
    keeps points up to ``i`` and from ``j`` onward and discards what is between.
    ``step`` is the typical writing step from :func:`typical_step`; 0 means the
    speed test is unavailable and only the conservative length rule applies.
    """
    if len(points) < 2 or height <= 0:
        return []

    limit = cfg.max_rise * height
    shortest = min(cfg.min_length, cfg.fast_min_length) * height
    runs: List[Tuple[int, int]] = []
    start = 0
    while start < len(points) - 1:
        end = _low_rise_end(points, start, limit)
        # Low-rise runs through writing are short, so the cheap length test
        # rejects almost every position here and the scan stays near-linear.
        if end > start and math.dist(points[start], points[end]) >= shortest:
            interior = _is_writing(points[: start + 1], height, cfg) and _is_writing(
                points[end:], height, cfg
            )
            if _is_traverse(
                points[start : end + 1], height, step, cfg, interior=interior
            ):
                runs.append((start, end))
                start = end + 1
                continue
        start += 1
    return runs


def _diagonal(points: Sequence[Point]) -> float:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _is_writing(points: Sequence[Point], height: float, cfg: CleanupConfig) -> bool:
    """Is a fragment left over from a cut big enough to be part of a letter?"""
    return len(points) >= 2 and _diagonal(points) >= cfg.min_fragment * height


def _split_stroke(
    stroke: Stroke, height: float, step: float, cfg: CleanupConfig
) -> Tuple[List[Stroke], int, float]:
    """Cut the traverses out of one stroke. Returns (pieces, cuts, pixels cut)."""
    points = stroke.points
    runs = find_traverses(points, height, cfg, step)
    if not runs:
        return [stroke], 0, 0.0

    pieces: List[Stroke] = []
    removed = 0.0
    cursor = 0
    for i, j in runs:
        head = points[cursor : i + 1]
        if _is_writing(head, height, cfg):
            pieces.append(Stroke(list(head)))
        # Arc, not chord: the report is compared against the drawn length of
        # the whole sample, which is also an arc. Mixing the two understates it.
        removed += sum(_steps(points[i : j + 1]))
        cursor = j
    tail = points[cursor:]
    if _is_writing(tail, height, cfg):
        pieces.append(Stroke(list(tail)))
    return pieces, len(runs), removed


def _x_gap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Horizontal distance between two x-intervals; 0 when they overlap."""
    return max(0.0, max(a[0], b[0]) - min(a[1], b[1]))


def _drop_stray_specks(
    strokes: Sequence[Stroke], height: float, cfg: CleanupConfig
) -> Tuple[List[Stroke], int]:
    """Remove tiny marks that sit away from the writing.

    Small strokes are mostly legitimate — the dot on an i, a full stop, a
    comma — so size alone is never enough. A speck is only dropped when it is
    also *detached*: horizontally clear of every other stroke, or sitting well
    above or below the band the real writing occupies.
    """
    if len(strokes) < 2:
        return list(strokes), 0

    spans = [
        (min(x for x, _ in s.points), max(x for x, _ in s.points)) for s in strokes
    ]
    specks = [_diagonal(s.points) <= cfg.speck_size * height for s in strokes]

    body = [s for s, speck in zip(strokes, specks) if not speck]
    if body:
        ys = [y for s in body for _, y in s.points]
        band_top = min(ys) - cfg.speck_drift * height
        band_bottom = max(ys) + cfg.speck_drift * height
    else:
        band_top, band_bottom = -math.inf, math.inf

    kept: List[Stroke] = []
    dropped = 0
    for index, stroke in enumerate(strokes):
        if not specks[index]:
            kept.append(stroke)
            continue
        isolated = all(
            _x_gap(spans[index], spans[other]) > cfg.speck_isolation * height
            for other in range(len(strokes))
            if other != index
        )
        ys = [y for _, y in stroke.points]
        drifted = min(ys) > band_bottom or max(ys) < band_top
        if isolated or drifted:
            dropped += 1
            continue
        kept.append(stroke)
    return kept, dropped


def clean_ink(
    ink: Ink, cfg: Optional[CleanupConfig] = None
) -> Tuple[Ink, CleanupReport]:
    """Return ``ink`` with drags, no-lift connectors and stray marks removed.

    The cleaned ink is a new :class:`Ink`; strokes that were not cut are shared
    with the input rather than copied. When cleanup would leave nothing behind
    the original is returned untouched — a blank pad is never an improvement on
    a messy one, and the report says so.
    """
    cfg = cfg or CleanupConfig()
    live = [s for s in ink.strokes if len(s) > 0]
    if not live:
        return ink, CleanupReport(
            strokes_in=len(ink.strokes), strokes_out=len(ink.strokes)
        )

    height = writing_height(ink)
    step = typical_step(ink)
    pieces: List[Stroke] = []
    cuts = 0
    pixels = 0.0
    for stroke in live:
        split, count, removed = _split_stroke(stroke, height, step, cfg)
        pieces.extend(split)
        cuts += count
        pixels += removed

    pieces, specks = _drop_stray_specks(pieces, height, cfg)

    if not pieces:
        return ink, CleanupReport(
            height=height,
            strokes_in=len(live),
            strokes_out=len(live),
            reverted=True,
        )

    return Ink(pieces), CleanupReport(
        height=height,
        strokes_in=len(live),
        strokes_out=len(pieces),
        traverses_cut=cuts,
        traverse_pixels=pixels,
        specks_dropped=specks,
    )
