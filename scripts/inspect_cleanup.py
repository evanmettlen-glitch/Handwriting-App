"""Check what ink cleanup does to your real samples before trusting it.

Cleanup deletes strokes. The thresholds in :mod:`handwriting_app.cleanup` were
chosen on synthetic ink, so the only honest way to accept them is to run them
over handwriting you actually wrote and look at two numbers:

* **word counts** — cleanup should make ``segment_words`` agree with the label
  more often, because cutting the slide between two words puts the pen lift
  back. This is the number it is supposed to move.
* **ink removed** — the share of drawn length it deleted. On a sample written
  with clean pen lifts this should be 0%. A real drag can be a big share, so
  what to worry about is ink deleted on a sample whose word count did not
  improve: that is a letter going missing, and no word-count win excuses it.

    python -m scripts.inspect_cleanup
    python -m scripts.inspect_cleanup --sweep          # tune --min-length
    python -m scripts.inspect_cleanup --dump-png out/  # before/after renders

Needs no torch — runs in seconds.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

from handwriting_app.cleanup import CleanupConfig, clean_ink
from handwriting_app.dataset import iter_samples
from handwriting_app.ink import Ink
from handwriting_app.segmentation import segment_words

# Above this share of ink deleted on a sample *without* fixing its word count,
# cleanup is not tidying, it is damaging. A genuine drag can be a large share of
# the drawn length, so the share alone proves nothing — it has to have bought
# nothing as well.
ALARM_SHARE = 0.15


def drawn_length(ink: Ink) -> float:
    return sum(
        math.dist(a, b)
        for stroke in ink.strokes
        for a, b in zip(stroke.points, stroke.points[1:])
    )


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
        help="Word-split threshold used when scoring word counts (default: 0.4).",
    )
    p.add_argument(
        "--min-length",
        type=float,
        default=CleanupConfig.min_length,
        help="Traverse length that counts as a drag, in writing heights "
        f"(default: {CleanupConfig.min_length}).",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Try a range of --min-length values and report which one is best.",
    )
    return p.parse_args()


def score(samples, cfg: CleanupConfig, gap_ratio: float) -> tuple[int, int, float]:
    """(word counts right before, right after, worst share of ink removed)."""
    before = after = 0
    worst = 0.0
    for sample in samples:
        want = len(sample.label.split())
        if len(segment_words(sample.ink, gap_ratio=gap_ratio)) == want:
            before += 1
        cleaned, report = clean_ink(sample.ink, cfg)
        if len(segment_words(cleaned, gap_ratio=gap_ratio)) == want:
            after += 1
        total = drawn_length(sample.ink)
        if total:
            worst = max(worst, report.traverse_pixels / total)
    return before, after, worst


def main() -> None:
    args = parse_args()
    samples = list(iter_samples(args.samples))
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No samples in {args.samples!r}")

    cfg = replace(CleanupConfig(), min_length=args.min_length)

    print(f"{len(samples)} samples from {args.samples}\n")
    print(
        f"{'label':<28} {'strokes':>9} {'drags':>6} {'specks':>7} "
        f"{'ink cut':>8} {'words':>12}"
    )
    print("-" * 78)

    changed = touched_clean = 0
    cut_pixels = total_pixels = 0.0
    before_right = after_right = 0

    for sample in samples:
        want = len(sample.label.split())
        was = len(segment_words(sample.ink, gap_ratio=args.gap_ratio))
        cleaned, report = clean_ink(sample.ink, cfg)
        now = len(segment_words(cleaned, gap_ratio=args.gap_ratio))

        before_right += was == want
        after_right += now == want
        drawn = drawn_length(sample.ink)
        total_pixels += drawn
        cut_pixels += report.traverse_pixels
        share = report.traverse_pixels / drawn if drawn else 0.0

        flags = []
        if report.changed:
            changed += 1
        helped = now == want and was != want
        if share > ALARM_SHARE and not helped:
            flags.append("EATING INK")
            touched_clean += 1
        if now != was:
            flags.append("words " + ("fixed" if now == want else "changed"))
        if report.reverted:
            flags.append("reverted")
        flag = ("  <-- " + ", ".join(flags)) if flags else ""

        label = sample.label if len(sample.label) <= 27 else sample.label[:24] + "..."
        print(
            f"{label:<28} {report.strokes_in:>4}->{report.strokes_out:<4} "
            f"{report.traverses_cut:>6} {report.specks_dropped:>7} "
            f"{share:>7.1%} {was}->{now}/{want:<6}{flag}"
        )

    share = cut_pixels / total_pixels if total_pixels else 0.0
    print("\n" + "=" * 78)
    print(f"samples changed  {changed}/{len(samples)}")
    print(f"ink removed      {share:.2%} of total drawn length")
    print(
        f"word counts      {before_right}/{len(samples)} right before, "
        f"{after_right}/{len(samples)} after"
    )

    if args.sweep:
        print("\nmin-length sweep (word counts right, higher is better):")
        rows = []
        for length in (0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0):
            _, right, worst = score(
                samples, replace(cfg, min_length=length), args.gap_ratio
            )
            rows.append((length, right, worst))
        best = max(right for _, right, _ in rows)
        # Prefer the longest threshold that ties: a longer traverse is a safer
        # thing to delete, so buy accuracy at the lowest risk that gets it.
        pick = max(length for length, right, _ in rows if right == best)
        for length, right, worst in rows:
            mark = "  <-- pick" if length == pick else ""
            print(
                f"  {length:>4}   {right:>3}/{len(samples)} right   "
                f"worst sample lost {worst:>5.1%} of its ink{mark}"
            )
        print(
            f"\n  best: --min-length {pick} "
            f"({best}/{len(samples)} right). To keep it, set that as the default "
            "of CleanupConfig.min_length."
        )

    print()
    if touched_clean:
        print(
            f"VERDICT: cleanup is DELETING WRITING on {touched_clean} sample(s) — more\n"
            f"than {ALARM_SHARE:.0%} of the drawn length went and bought nothing.\n"
            "Dump the renders and look:\n"
            "  python -m scripts.inspect_cleanup --dump-png out/\n"
            "If the cut ink is really part of a letter, raise CleanupConfig.min_length\n"
            "or run the app with --no-cleanup until it is fixed.\n"
        )
    elif after_right > before_right:
        print(
            f"VERDICT: cleanup FIXED {after_right - before_right} sample(s)' word counts\n"
            f"and removed {share:.1%} of the ink doing it. That is the case it exists for —\n"
            "writing without lifting the pen between words.\n"
        )
    elif changed == 0:
        print(
            "VERDICT: cleanup changed NOTHING on this set, which is the right answer\n"
            "for samples written with clean pen lifts. Collect a few written without\n"
            "lifting, or with a deliberate finger drag across the pad, to exercise it:\n"
            "  ./run.sh --train\n"
        )
    else:
        print(
            f"VERDICT: cleanup removed {share:.1%} of the ink without moving word counts.\n"
            "Harmless, but check a dump to confirm it took drags and not letters.\n"
        )

    if args.dump_png:
        out = Path(args.dump_png)
        out.mkdir(parents=True, exist_ok=True)
        written = 0
        for index, sample in enumerate(samples, 1):
            cleaned, report = clean_ink(sample.ink, cfg)
            if not report.changed:
                continue
            for name, ink in (("before", sample.ink), ("after", cleaned)):
                image = ink.render(stroke_width=sample.stroke_width, deslant=True)
                if image is not None:
                    image.save(out / f"{index:04d}-{name}.png")
            written += 1
        print(f"Wrote {written} before/after pairs to {out}/ — open them and look.")


if __name__ == "__main__":
    main()
