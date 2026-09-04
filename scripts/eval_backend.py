"""Measure how well a backend reads a folder of labelled samples.

This is the number every accuracy decision should be judged against. One pass,
no tuning — unlike calibrate.py, which runs several.

    python -m scripts.eval_backend                          # default backend
    python -m scripts.eval_backend --backend tesseract
    python -m scripts.eval_backend --model-dir models/evan
    python -m scripts.eval_backend --limit 10               # quick smoke check
                                                              #  (spread across
                                                              #   the set, not
                                                              #   the easiest 10)

Reports overall CER and exact-match word accuracy, split into natural-language
prompts and rote coverage prompts (alphabet runs, digit strings) — a
language-model decoder mangles the latter however neatly they were written, so
mixing them into one score is misleading.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from handwriting_app.calibration import load as load_calibration
from handwriting_app.config import AppConfig
from handwriting_app.dataset import iter_samples, representative
from handwriting_app.enrollment import is_rote
from handwriting_app.lexicon import personal_word_counts
from handwriting_app.pipeline import (
    PipelineConfig,
    RecognitionPipeline,
    resolve_segment,
)
from handwriting_app.recognizer import build_recognizer
from handwriting_app.textalign import cer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--samples", default="data/samples")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "tesseract", "trocr", "trocr-torch", "trocr-onnx"],
    )
    p.add_argument("--model-dir", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-spellcheck", dest="spellcheck", action="store_false")
    p.add_argument(
        "--no-lexicon",
        dest="lexicon",
        action="store_false",
        help="Don't feed the personal word list to the corrector.",
    )
    p.add_argument(
        "--no-calibration",
        dest="calibration",
        action="store_false",
        help="Ignore calibration.json even if present.",
    )
    p.add_argument(
        "--no-segment",
        dest="segment",
        action="store_const",
        const=False,
        default=None,
        help="Recognize whole lines (the default for TrOCR).",
    )
    p.add_argument(
        "--segment", dest="segment", action="store_const", const=True,
        help="Force word-by-word recognition.",
    )
    p.add_argument("--quiet", action="store_true", help="Summary only.")
    return p.parse_args()


def summarize(rows: List[Tuple[str, str, float]], title: str) -> None:
    if not rows:
        return
    mean_cer = sum(r[2] for r in rows) / len(rows)
    exact = sum(1 for gold, pred, _ in rows if pred == gold)
    print(
        f"{title:<24} n={len(rows):<4} CER {mean_cer:.3f}   "
        f"exact {exact}/{len(rows)} ({exact / len(rows):.0%})"
    )


def main() -> None:
    args = parse_args()

    everything = list(iter_samples(args.samples))
    samples = representative(everything, args.limit)
    if not samples:
        raise SystemExit(
            f"No samples in {args.samples!r}. Collect some with ./run.sh --train"
        )
    if args.limit and args.limit < len(everything):
        scored = sum(1 for s in samples if not is_rote(s.label))
        print(
            f"--limit {args.limit} of {len(everything)}: spread across the set, "
            f"not the first {args.limit} — {scored} will count toward CER below.\n"
            "For a real accuracy number, drop --limit and use the full set.\n"
        )

    config = AppConfig(backend=args.backend, model_dir=args.model_dir)
    recognizer = build_recognizer(config)

    calibration = load_calibration(args.samples) if args.calibration else None
    lexicon = dict(personal_word_counts(args.samples)) if args.lexicon else {}
    pipeline = RecognitionPipeline(
        recognizer,
        PipelineConfig(
            segment=resolve_segment(args.segment, recognizer.name),
            spellcheck=args.spellcheck,
            personal_lexicon=lexicon,
            calibration=calibration,
        ),
    )

    print(f"{len(samples)} samples from {args.samples}")
    print(f"recognizer: {recognizer.name}")
    for note in pipeline.notes:
        print(f"  · {note}")
    print()

    natural: List[Tuple[str, str, float]] = []
    rote: List[Tuple[str, str, float]] = []
    for index, sample in enumerate(samples, 1):
        pred = pipeline.run(sample.ink).strip()
        score = cer(pred, sample.label)
        (rote if is_rote(sample.label) else natural).append(
            (sample.label, pred, score)
        )
        if not args.quiet:
            flag = "rote" if is_rote(sample.label) else "    "
            print(f"{index:>3}/{len(samples)} {flag} {score:.2f}  {sample.label!r} -> {pred!r}")

    print()
    summarize(natural, "natural language")
    summarize(rote, "rote coverage")
    summarize(natural + rote, "ALL")

    if natural:
        worst = sorted(natural, key=lambda r: -r[2])[:5]
        print("\nworst natural-language samples:")
        for gold, pred, score in worst:
            print(f"  {score:.2f}  {gold!r} -> {pred!r}")


if __name__ == "__main__":
    main()
