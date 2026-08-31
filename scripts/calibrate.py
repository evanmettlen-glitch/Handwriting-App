"""Personalize without training: one forward pass over the collected samples.

Instead of fine-tuning weights (which needs hundreds of samples), this measures
how the stock recognizer does on *your* handwriting and adapts around it:

  1. tries a handful of render settings, keeps the one with the lowest CER
  2. records whole words the recognizer reliably gets wrong for you
  3. writes data/samples/calibration.json, which the app loads at startup

Takes minutes and works with as few as ~20 samples.

    python -m scripts.calibrate                      # data/samples
    python -m scripts.calibrate --samples data/samples/evan
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

from handwriting_app.calibration import Calibration, save
from handwriting_app.config import AppConfig
from handwriting_app.dataset import iter_samples
from handwriting_app.recognizer import build_recognizer
from handwriting_app.segmentation import segment_words
from handwriting_app.textalign import cer

# (deslant, stroke_width, pad) combinations worth trying.
RENDER_GRID = [
    (True, 8, 32),
    (False, 8, 32),
    (True, 6, 32),
    (True, 10, 40),
    (True, 8, 16),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--backend", default="auto", choices=["auto", "tesseract", "trocr"])
    p.add_argument("--model-dir", default="")
    p.add_argument("--limit", type=int, default=0, help="Only use the first N samples.")
    p.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Times a word must be misread the same way to become a fix (default: 2).",
    )
    p.add_argument("--no-segment", dest="segment", action="store_false")
    return p.parse_args()


def recognize(recognizer, ink, *, deslant, stroke_width, pad, segment):
    words = segment_words(ink) if segment else [ink]
    if not words:
        words = [ink]
    pieces = []
    for word in words:
        image = word.render(stroke_width=stroke_width, pad=pad, deslant=deslant)
        if image is None:
            continue
        text = recognizer.recognize(image, hint="word" if segment else "line").strip()
        if text:
            pieces.append(text)
    return " ".join(pieces)


def main() -> None:
    args = parse_args()

    samples = list(iter_samples(args.samples))
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(
            f"No samples in {args.samples!r}. Collect some with ./run.sh --train"
        )
    print(f"{len(samples)} samples from {args.samples}")

    recognizer = build_recognizer(
        AppConfig(backend=args.backend, model_dir=args.model_dir)
    )
    print(f"recognizer: {recognizer.name}\n")

    # --- 1. pick the best render settings -----------------------------------
    best = None
    baseline = None
    for deslant, width, pad in RENDER_GRID:
        total = 0.0
        for sample in samples:
            pred = recognize(
                recognizer, sample.ink,
                deslant=deslant, stroke_width=width, pad=pad, segment=args.segment,
            )
            total += cer(pred, sample.label)
        score = total / len(samples)
        label = f"deslant={deslant} width={width} pad={pad}"
        print(f"  CER {score:.3f}   {label}")
        if baseline is None:
            baseline = score
        if best is None or score < best[0]:
            best = (score, deslant, width, pad)

    score, deslant, width, pad = best
    print(f"\nbest: CER {score:.3f}  (deslant={deslant} width={width} pad={pad})")

    # --- 2. mine reliable whole-word fixes ----------------------------------
    misread: Dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        pred = recognize(
            recognizer, sample.ink,
            deslant=deslant, stroke_width=width, pad=pad, segment=args.segment,
        )
        got = pred.split()
        want = sample.label.split()
        if len(got) != len(want):
            continue  # only trust word-for-word alignments
        for g, w in zip(got, want):
            if g and g != w:
                misread[g][w] += 1

    fixes = {
        wrong: counts.most_common(1)[0][0]
        for wrong, counts in misread.items()
        if counts.most_common(1)[0][1] >= args.min_occurrences
    }
    print(f"{len(fixes)} reliable word fixes" + (f": {fixes}" if fixes else ""))

    # --- 3. score with fixes applied ---------------------------------------
    calibration = Calibration(
        deslant=deslant, stroke_width=width, render_pad=pad,
        fixes=fixes, baseline_cer=round(baseline, 4), samples=len(samples),
    )
    total = 0.0
    for sample in samples:
        pred = recognize(
            recognizer, sample.ink,
            deslant=deslant, stroke_width=width, pad=pad, segment=args.segment,
        )
        total += cer(calibration.apply_fixes(pred), sample.label)
    calibration.tuned_cer = round(total / len(samples), 4)

    path = save(calibration, args.samples)
    print(
        f"\nCER {calibration.baseline_cer:.3f} -> {calibration.tuned_cer:.3f}"
        f"\nWrote {path}. The app loads it automatically."
    )


if __name__ == "__main__":
    main()
