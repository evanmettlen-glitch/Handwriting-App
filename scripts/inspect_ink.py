"""Diagnose stroke capture quality — is the touchscreen giving us enough points?

Tk coalesces motion events, so a fast stroke can be recorded as a handful of
widely-spaced points. Rendering joins them with straight lines, which turns
letters into angular polygons that no handwriting model can read.

    python -m scripts.inspect_ink
    python -m scripts.inspect_ink --dump-png out/     # eyeball the renders

Needs no torch — runs in seconds.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median

from handwriting_app.dataset import iter_samples

# Above this, straight-line rendering visibly corners. Handwriting strokes want
# points every few pixels.
SPARSE_PX = 6.0


def spacings(stroke) -> list[float]:
    pts = stroke.points
    return [
        math.dist(a, b)
        for a, b in zip(pts, pts[1:])
        if math.dist(a, b) > 0
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dump-png", metavar="DIR", default="")
    p.add_argument(
        "--no-smooth",
        dest="smooth",
        action="store_false",
        help="Render the dumped PNGs without spline smoothing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = list(iter_samples(args.samples))
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No samples in {args.samples!r}")

    all_gaps: list[float] = []
    total_points = 0
    total_strokes = 0
    widths: list[float] = []

    print(f"{len(samples)} samples from {args.samples}\n")
    print(f"{'label':<28} {'strokes':>7} {'points':>7} {'med gap':>8} {'width':>7}")
    print("-" * 62)

    for sample in samples:
        gaps = [g for s in sample.ink.strokes for g in spacings(s)]
        points = sum(len(s) for s in sample.ink.strokes)
        bounds = sample.ink.bounds()
        width = (bounds[2] - bounds[0]) if bounds else 0.0
        med = median(gaps) if gaps else 0.0

        all_gaps.extend(gaps)
        total_points += points
        total_strokes += len(sample.ink.strokes)
        widths.append(width)

        flag = "  <-- sparse" if med > SPARSE_PX else ""
        label = sample.label if len(sample.label) <= 27 else sample.label[:24] + "..."
        print(f"{label:<28} {len(sample.ink.strokes):>7} {points:>7} {med:>8.1f} {width:>7.0f}{flag}")

    overall = median(all_gaps) if all_gaps else 0.0
    sparse = sum(
        1
        for s in samples
        if (g := [x for st in s.ink.strokes for x in spacings(st)]) and median(g) > SPARSE_PX
    )

    print("\n" + "=" * 62)
    print(f"strokes/sample   {total_strokes / len(samples):.1f}")
    print(f"points/sample    {total_points / len(samples):.0f}")
    print(f"median gap       {overall:.1f} px")
    print(f"median width     {median(widths):.0f} px")
    print(f"sparse samples   {sparse}/{len(samples)} (median gap > {SPARSE_PX:.0f} px)")

    print()
    if overall > SPARSE_PX:
        print(
            "VERDICT: capture is UNDER-SAMPLED. Tk is coalescing motion events, so\n"
            "strokes are stored as widely-spaced points and render as straight-line\n"
            "polygons. Spline smoothing in Ink.render(smooth=True) reconstructs the\n"
            "curve — compare with:\n"
            "  python -m scripts.eval_backend --limit 10\n"
        )
    else:
        print(
            "VERDICT: capture density looks fine. Poor accuracy is not coming from\n"
            "sparse sampling — look at the per-sample predictions instead:\n"
            "  python -m scripts.eval_backend\n"
        )

    if args.dump_png:
        out = Path(args.dump_png)
        out.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples, 1):
            image = sample.ink.render(
                stroke_width=sample.stroke_width, deslant=True, smooth=args.smooth
            )
            if image is not None:
                image.save(out / f"{index:04d}.png")
        print(f"Wrote {len(samples)} renders to {out}/ — open a few and look at them.")


if __name__ == "__main__":
    main()
