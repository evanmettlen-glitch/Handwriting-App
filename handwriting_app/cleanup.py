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
that carries no vertical information. Letters mostly do not do that — but a
wide cursive letter can get close: the flat top of an 'e' bowl or a 't' crossbar
measured at 1.0-1.1x the writing height on real handwriting (see below), well
past where synthetic test letters ever reached. A traverse has to clear a
length threshold with real margin above that, or fewer if it was also covered
fast enough to have been a move rather than a mark. So one detector handles
drags and missing pen lifts alike — cut the traverse out and the strokes on
either side become separate words again.

Everything is measured as a fraction of the writing height, so the thresholds
hold for big and small handwriting alike. Nothing here mutates the input: split
strokes are new objects and untouched strokes are shared, matching what
:mod:`handwriting_app.segmentation` already does.

**Measured on real handwriting, 2026-09-03.** The first version of this module
was tuned on synthetic ink alone and shipped with ``min_length=1.0``. Run
against the 43 real enrollment samples on the Pi, it visibly destroyed letters
in 2 of them — the flat top of the 'e' in "the" and the 't' crossbar in "they"
both measured 1.02-1.08x the writing height, comfortably clearing what had been
the safe threshold. ``python -m scripts.inspect_cleanup --sweep`` on that same
data found 1.0-1.0 unsafe (worst case lost 15.8% of a sample's ink for no
benefit), 1.2 the first value with zero loss, and 1.5-3.0 tied with it exactly —
so the default here is 1.5, a margin above the first safe point rather than the
sweep's own longest-tied pick, which would blunt sensitivity to genuine drags
for no measured gain. ``fast_min_length`` moved by the same ratio for the same
reason, though it is unexercised by this dataset (none of the 43 samples were
written without lifting the pen) and still wants real no-lift data to confirm.
"""

from __future__ import annotations

import math
from collections import deque
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

    #: End-to-end length a traverse must reach to count as one. 1.0 measured
    #: unsafe on real cursive (see the module docstring) — keep a margin above
    #: 1.2, the first value that scored zero false positives on real data.
    min_length: float = 1.5
    #: Length that suffices when the move was also fast (see ``fast_ratio``).
    #: Scaled from ``min_length`` by the same ratio as before the correction
    #: above; unexercised by the real dataset that motivated it.
    fast_min_length: float = 0.75
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

    Reference implementation, kept because it is obviously correct and is the
    oracle :func:`_low_rise_ends` is property-tested against.
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


def _low_rise_ends(points: Sequence[Point], limit: float) -> List[int]:
    """:func:`_low_rise_end` for every start index, in one linear pass.

    Calling the reference version from each start is quadratic whenever the
    windows are long, which happens exactly when a stroke contains a long flat
    stretch — a scribbled-out word, a strikethrough, a slow underline. Measured
    before this existed: a 4000-point stroke of that shape took **1.06 s** of
    pure geometry, against a documented budget of "sub-millisecond".

    The window end is monotone in ``start`` — dropping a point off the left can
    only let the window reach further right — so one forward-only sweep does
    it, with monotonic deques carrying the running min and max of ``y`` across
    the window (each index is pushed and popped at most once).
    """
    count = len(points)
    ends: List[int] = [0] * count
    lows: deque = deque()  # indices, y increasing  -> front is the window minimum
    highs: deque = deque()  # indices, y decreasing -> front is the window maximum
    end = -1

    for start in range(count):
        if end < start:
            # Window is empty: restart it on `start` itself.
            end = start
            lows.clear()
            highs.clear()
            lows.append(start)
            highs.append(start)
        while end + 1 < count:
            y = points[end + 1][1]
            low = min(points[lows[0]][1], y)
            high = max(points[highs[0]][1], y)
            if high - low > limit:
                break
            end += 1
            while lows and points[lows[-1]][1] >= y:
                lows.pop()
            lows.append(end)
            while highs and points[highs[-1]][1] <= y:
                highs.pop()
            highs.append(end)
        ends[start] = end
        if lows and lows[0] == start:
            lows.popleft()
        if highs and highs[0] == start:
            highs.popleft()
    return ends


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


def _prefix_abs_dx(points: Sequence[Point]) -> List[float]:
    """Cumulative horizontal distance travelled, so any range is one subtraction."""
    out = [0.0] * len(points)
    for i in range(1, len(points)):
        out[i] = out[i - 1] + abs(points[i][0] - points[i - 1][0])
    return out


def _prefix_reaches(points: Sequence[Point], size: float) -> List[bool]:
    """Per index i: is ``points[:i+1]`` big enough to be writing rather than a nub?"""
    out = [False] * len(points)
    x0 = x1 = points[0][0]
    y0 = y1 = points[0][1]
    for i in range(1, len(points)):
        x, y = points[i]
        x0, x1 = min(x0, x), max(x1, x)
        y0, y1 = min(y0, y), max(y1, y)
        out[i] = math.hypot(x1 - x0, y1 - y0) >= size
    return out


def _suffix_reaches(points: Sequence[Point], size: float) -> List[bool]:
    """Per index i: is ``points[i:]`` big enough to be writing rather than a nub?"""
    count = len(points)
    out = [False] * count
    last = count - 1
    x0 = x1 = points[last][0]
    y0 = y1 = points[last][1]
    for i in range(count - 2, -1, -1):
        x, y = points[i]
        x0, x1 = min(x0, x), max(x1, x)
        y0, y1 = min(y0, y), max(y1, y)
        out[i] = math.hypot(x1 - x0, y1 - y0) >= size
    return out


def _is_traverse(
    points: Sequence[Point],
    a: int,
    b: int,
    height: float,
    step: float,
    cfg: CleanupConfig,
    absdx: Sequence[float],
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
    chord = math.dist(points[a], points[b])
    if chord < min(cfg.fast_min_length, cfg.min_length) * height:
        return False

    # Straightness in 2-D would be the obvious second test and it double-counts
    # the vertical wander the rise limit already caps — on a short traverse the
    # curl of the letters at either end then sinks it. What is left to check is
    # that the pen went one way: a scribbled-out word stays just as flat.
    #
    # Done before the stride scan because this is the O(1) one and it is what a
    # scribble fails. Order matters for speed, not for the answer.
    travelled = absdx[b] - absdx[a]
    if travelled <= 0:
        return False
    if abs(points[b][0] - points[a][0]) / travelled < cfg.min_directness:
        return False

    # The only scan left, and it runs just for runs that already look like a
    # traverse — those are disjoint, so the total stays linear.
    strides = _steps(points[a : b + 1])
    fast = step > 0 and strides and max(strides) >= cfg.fast_ratio * step
    lenient = fast and interior
    return chord >= (cfg.fast_min_length if lenient else cfg.min_length) * height


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
    ends = _low_rise_ends(points, limit)

    # Everything below has to be answerable in constant time per start, or the
    # scan degrades to quadratic on exactly the input it most needs to survive:
    # a long flat non-one-way run (a scribbled-out word, a strikethrough), where
    # the window is long at *every* start and nothing short-circuits. Measured
    # before these were precomputed, a 4000-point stroke of that shape cost
    # 1.06 s of pure geometry.
    fragment = cfg.min_fragment * height
    head_big = _prefix_reaches(points, fragment)
    tail_big = _suffix_reaches(points, fragment)
    absdx = _prefix_abs_dx(points)

    runs: List[Tuple[int, int]] = []
    start = 0
    while start < len(points) - 1:
        end = ends[start]
        if end > start and math.dist(points[start], points[end]) >= shortest:
            interior = head_big[start] and tail_big[end]
            if _is_traverse(
                points, start, end, height, step, cfg, absdx, interior=interior
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
