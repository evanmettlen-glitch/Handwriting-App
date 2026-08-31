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
from handwriting_app.segmentation import segment_words

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
        "--gap-ratio",
        type=float,
        default=0.4,
        help="Word-split threshold to test (default: 0.4, the app's default).",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Try a range of --gap-ratio values and report which segments best.",
    )
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

    seg_wrong = split = merged = 0
    print(f"{len(samples)} samples from {args.samples}\n")
    print(
        f"{'label':<28} {'strokes':>7} {'points':>7} {'med gap':>8} "
        f"{'width':>7} {'words':>11}"
    )
    print("-" * 76)

    for sample in samples:
        gaps = [g for s in sample.ink.strokes for g in spacings(s)]
        points = sum(len(s) for s in sample.ink.strokes)
        bounds = sample.ink.bounds()
        width = (bounds[2] - bounds[0]) if bounds else 0.0
        med = median(gaps) if gaps else 0.0

        want_words = len(sample.label.split())
        got_words = len(segment_words(sample.ink, gap_ratio=args.gap_ratio))
        if got_words != want_words:
            seg_wrong += 1
            if got_words > want_words:
                split += 1
            else:
                merged += 1

        all_gaps.extend(gaps)
        total_points += points
        total_strokes += len(sample.ink.strokes)
        widths.append(width)

        flags = []
        if med > SPARSE_PX:
            flags.append("sparse")
        if got_words != want_words:
            flags.append("SPLIT" if got_words > want_words else "merged")
        flag = ("  <-- " + ",".join(flags)) if flags else ""

        label = sample.label if len(sample.label) <= 27 else sample.label[:24] + "..."
        print(
            f"{label:<28} {len(sample.ink.strokes):>7} {points:>7} {med:>8.1f} "
            f"{width:>7.0f} {got_words:>4}/{want_words:<6}{flag}"
        )

    overall = median(all_gaps) if all_gaps else 0.0
    sparse = sum(
        1
        for s in samples
        if (g := [x for st in s.ink.strokes for x in spacings(st)]) and median(g) > SPARSE_PX
    )

    print("\n" + "=" * 76)
    print(f"strokes/sample   {total_strokes / len(samples):.1f}")
    print(f"points/sample    {total_points / len(samples):.0f}")
    print(f"median gap       {overall:.1f} px")
    print(f"median width     {median(widths):.0f} px")
    print(f"sparse samples   {sparse}/{len(samples)} (median gap > {SPARSE_PX:.0f} px)")
    print(
        f"mis-segmented    {seg_wrong}/{len(samples)} at --gap-ratio "
        f"{args.gap_ratio}  ({split} over-split, {merged} merged)"
    )

    if args.sweep:
        print("\ngap-ratio sweep (wrong word counts, lower is better):")
        scores = []
        for ratio in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0):
            wrong = sum(
                1
                for s in samples
                if len(segment_words(s.ink, gap_ratio=ratio)) != len(s.label.split())
            )
            scores.append((ratio, wrong))

        fewest = min(wrong for _, wrong in scores)
        tied = [ratio for ratio, wrong in scores if wrong == fewest]
        # Prefer the middle of the tied range: a threshold sitting next to a
        # failure is one sloppy sample away from breaking.
        best = tied[len(tied) // 2]

        for ratio, wrong in scores:
            mark = "  <-- pick" if ratio == best else (" *" if wrong == fewest else "")
            print(f"  {ratio:>4}   {wrong:>3}/{len(samples)} wrong{mark}")
        print(
            f"\n  best: --word-gap-ratio {best} "
            f"({fewest}/{len(samples)} wrong; ties {tied[0]}-{tied[-1]})"
        )

    print()
    if split > len(samples) * 0.15:
        print(
            "VERDICT: words are being OVER-SPLIT — the recognizer is handed letter\n"
            "fragments instead of words, which produces random errors immune to\n"
            "render tuning. Raise the threshold:\n"
            "  ./run.sh --word-gap-ratio <best above>\n"
        )
    elif merged > len(samples) * 0.25:
        print(
            "VERDICT: words are being MERGED into whole lines.\n"
            "For TrOCR this is FINE — it was trained on IAM text lines and uses\n"
            "cross-word context, so a whole line is what it wants. Segmentation only\n"
            "helps tesseract. Run the line-at-a-time path (the default for TrOCR):\n"
            "  ./run.sh --no-segment\n"
            "If accuracy is still poor, the model is the limit, not the plumbing:\n"
            "  python -m scripts.eval_backend --no-segment\n"
        )
    elif overall > SPARSE_PX:
        print(
            "VERDICT: capture is UNDER-SAMPLED. Tk is coalescing motion events, so\n"
            "strokes are stored as widely-spaced points and render as straight-line\n"
            "polygons. Spline smoothing (Ink.render(smooth=True), on by default)\n"
            "reconstructs the curve — measure it with:\n"
            "  python -m scripts.calibrate\n"
        )
    else:
        print(
            "VERDICT: capture and segmentation both look OK. The model itself is\n"
            "the limit — look at what it actually predicts:\n"
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
