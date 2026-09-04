"""Where the seconds go, and what each speed knob costs in accuracy.

Run this on the Pi — timings from a laptop mean nothing here.

    python -m scripts.bench_latency                 # stage breakdown + sweep
    python -m scripts.bench_latency --limit 4       # faster, noisier
    python -m scripts.bench_latency --stages        # breakdown only, no sweep
    python -m scripts.bench_latency --models microsoft/trocr-base-handwritten \
                                             microsoft/trocr-small-handwritten

Two parts:

1. **Stage breakdown** - one sample, timed through render -> preprocess ->
   encoder -> decode -> postprocess, so it is obvious whether the cost is the
   vision encoder (fixed per image) or token generation (grows with the line).

2. **Sweep** - median seconds *and* CER for each (model, beams, int8) combo, on
   real samples. Speed is only worth having if the accuracy column holds, so the
   two are always reported together. Rote coverage prompts (``abcdefghijklm``)
   are excluded from CER the way eval_backend excludes them; they are still
   timed.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from typing import List, Optional, Sequence, Tuple

from handwriting_app.calibration import load as load_calibration
from handwriting_app.dataset import Sample, iter_samples
from handwriting_app.enrollment import is_rote
from handwriting_app.lexicon import personal_word_counts
from handwriting_app.models import resolve_model_dir
from handwriting_app.pipeline import PipelineConfig, RecognitionPipeline, resolve_segment
from handwriting_app.textalign import cer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--model-dir", default="", help="Override the auto-discovered model.")
    p.add_argument("--user", default="")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="REF",
        help="Sweep these models instead of just the resolved one. Local dirs or "
        "Hugging Face ids (they download on first use).",
    )
    p.add_argument(
        "--beams",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Beam widths to compare (default: 1 4).",
    )
    p.add_argument(
        "--quantize",
        nargs="+",
        default=["off", "on"],
        choices=["off", "on"],
        help="int8 settings to compare (default: off on).",
    )
    p.add_argument(
        "--image-sizes",
        nargs="+",
        type=int,
        default=[0],
        metavar="PX",
        help="Encoder input sizes to compare; 0 is the checkpoint's native 384. "
        "Cost is set by the patch count, so 224 is about a third of the encoder "
        "work (default: 0).",
    )
    p.add_argument("--limit", type=int, default=8, help="Samples per config (default: 8).")
    p.add_argument("--stages", action="store_true", help="Stage breakdown only.")
    p.add_argument("--no-stages", action="store_true", help="Sweep only.")
    return p.parse_args()


# -- hardware state -------------------------------------------------------
def report_hardware() -> None:
    """Throttling looks exactly like slow code, and no code change fixes it."""
    def vcgencmd(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["vcgencmd", *args], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    temp = vcgencmd("measure_temp")
    throttled = vcgencmd("get_throttled")
    clock = vcgencmd("measure_clock", "arm")
    if temp is None and throttled is None:
        print("hardware: vcgencmd unavailable (not a Pi?) — timings may not transfer\n")
        return

    bits = ""
    if throttled and "=" in throttled:
        value = int(throttled.split("=", 1)[1], 0)
        flags = []
        if value & 0x1:
            flags.append("UNDER-VOLTAGE NOW")
        if value & 0x4:
            flags.append("ARM FREQ CAPPED NOW")
        if value & 0x8:
            flags.append("THROTTLED NOW")
        if value & 0x10000:
            flags.append("under-voltage since boot")
        if value & 0x40000:
            flags.append("freq capped since boot")
        if value & 0x80000:
            flags.append("throttled since boot")
        bits = "  ".join(flags) if flags else "clean"

    mhz = ""
    if clock and "=" in clock:
        mhz = f"{int(clock.split('=', 1)[1]) / 1e6:.0f} MHz"
    print(f"hardware: {temp or '?'}  {mhz}  throttle: {bits or '?'}")
    if bits and "NOW" in bits:
        print("  !! throttling right now — fix power/cooling before trusting these numbers")
    print()


# -- building -------------------------------------------------------------
def build(
    model_ref: str,
    beams: int,
    quantize: bool,
    samples_dir: str,
    image_size: int = 0,
):
    from handwriting_app.recognizer.trocr_torch_recognizer import TrocrTorchRecognizer

    recognizer = TrocrTorchRecognizer(
        model_dir=model_ref,
        num_beams=beams,
        quantize=quantize,
        image_size=image_size,
    )
    pipeline = RecognitionPipeline(
        recognizer,
        PipelineConfig(
            segment=resolve_segment(None, recognizer.name),
            personal_lexicon=dict(personal_word_counts(samples_dir)),
            calibration=load_calibration(samples_dir),
        ),
    )
    return recognizer, pipeline


# -- part 1: stage breakdown ---------------------------------------------
def stage_breakdown(pipeline: RecognitionPipeline, sample: Sample) -> None:
    recognizer = pipeline.recognizer
    torch = recognizer._torch  # noqa: SLF001 - benchmark needs the raw model
    model = recognizer._model  # noqa: SLF001
    processor = recognizer._processor  # noqa: SLF001

    print(f"stage breakdown on {sample.label!r}")

    t0 = time.perf_counter()
    image = pipeline.render(sample.ink)
    t1 = time.perf_counter()
    if image is None:
        print("  (nothing to render)\n")
        return

    pixel_values = recognizer._pixels(image, recognizer.image_size)  # noqa: SLF001
    t2 = time.perf_counter()

    extra = recognizer._generate_kwargs()  # noqa: SLF001
    with torch.inference_mode():
        encoded = model.encoder(pixel_values, **extra)
        t3 = time.perf_counter()
        ids = model.generate(
            pixel_values, max_new_tokens=recognizer.max_new_tokens, **extra
        )
    t4 = time.perf_counter()

    raw = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    text = pipeline.postprocess(raw)
    t5 = time.perf_counter()

    tokens = int(ids.shape[-1])
    encode = t3 - t2
    generate = t4 - t3
    # generate() re-runs the encoder internally, so the decode-only share is
    # what is left after subtracting one encoder pass.
    decode = max(0.0, generate - encode)
    total = t5 - t0

    rows = [
        ("render ink -> image", t1 - t0, f"{image.width}x{image.height}px"),
        ("processor (resize/normalize)", t2 - t1, str(tuple(pixel_values.shape))),
        ("vision encoder", encode, f"{encoded.last_hidden_state.shape[1]} patches"),
        ("decode", decode, f"{tokens} tokens"),
        ("postprocess (join + spell)", t5 - t4, repr(text)),
    ]
    for label, seconds, note in rows:
        share = seconds / total * 100 if total else 0
        print(f"  {label:<30} {seconds:6.2f}s  {share:4.0f}%   {note}")
    print(f"  {'TOTAL':<30} {total:6.2f}s\n")


# -- part 2: sweep --------------------------------------------------------
def representative(samples: Sequence[Sample], limit: int) -> List[Sample]:
    """An evenly-spaced subset, rather than the first ``limit`` samples.

    ``iter_samples`` yields enrollment order, which is deliberately graded:
    rote coverage prompts first, then single words, then multi-word lines. So
    slicing the front takes the *easiest* samples in the set — and because rote
    prompts are excluded from CER, a small limit can leave the whole accuracy
    column resting on a handful of one-word samples.

    Measured 2026-09-03, and the reason this function exists: the old
    ``--limit 8`` scored CER on exactly four samples — 'the', 'and', 'you',
    'was' — while all 21 multi-word samples went unmeasured. That produced a
    confident "CER 0.000, no accuracy cost" for a model swap which, measured
    over the full set, actually cost 0.425 -> 0.486. Spacing the picks keeps a
    limited run honest about what it is averaging.
    """
    if not limit or limit >= len(samples):
        return list(samples)
    step = len(samples) / limit
    return [samples[min(len(samples) - 1, int(i * step))] for i in range(limit)]


def measure(
    pipeline: RecognitionPipeline, samples: Sequence[Sample]
) -> Tuple[List[float], Optional[float], int]:
    times: List[float] = []
    scores: List[float] = []
    for sample in samples:
        started = time.perf_counter()
        pred = pipeline.run(sample.ink).strip()
        times.append(time.perf_counter() - started)
        if not is_rote(sample.label):
            scores.append(cer(pred, sample.label))
    mean = statistics.mean(scores) if scores else None
    return times, mean, len(scores)


def main() -> None:
    args = parse_args()

    everything = list(iter_samples(args.samples))
    samples = representative(everything, args.limit)
    if not samples:
        raise SystemExit(
            f"No samples in {args.samples!r}. Collect some with ./run.sh --train"
        )
    scored = sum(1 for s in samples if not is_rote(s.label))

    report_hardware()

    default_model = resolve_model_dir(args.model_dir, args.user)
    models = args.models or [default_model]

    if not args.no_stages:
        recognizer, pipeline = build(models[0], beams=1, quantize=False, samples_dir=args.samples)
        warm = recognizer.warmup()
        print(f"model: {recognizer.name}   warmup {warm:.1f}s\n")
        stage_breakdown(pipeline, samples[0])
        if args.stages:
            return

    print(
        f"sweep over {len(samples)} of {len(everything)} samples "
        f"(median seconds per line)"
    )
    print(
        f"CER is averaged over the {scored} natural-language sample(s) in that "
        f"subset; rote prompts are timed but not scored."
    )
    if scored < 10:
        print(
            f"  !! {scored} scored sample(s) is too few to compare accuracy on — "
            "the CER column\n"
            "     below is indicative at best. Raise --limit (or drop it "
            "entirely) before\n"
            "     concluding anything about accuracy from this run."
        )
    print()
    header = (
        f"{'model':<30} {'beams':>5} {'int8':>5} {'px':>5} "
        f"{'median':>8} {'p90':>7} {'CER':>6}"
    )
    print(header)
    print("-" * len(header))

    for model_ref in models:
        for beams in args.beams:
            for quant in args.quantize:
                for size in args.image_sizes:
                    quantize = quant == "on"
                    short = model_ref.split("/")[-1][:30]
                    try:
                        recognizer, pipeline = build(
                            model_ref, beams, quantize, args.samples, size
                        )
                    except Exception as exc:  # noqa: BLE001 - keep the sweep going
                        print(f"{short:<30} {beams:>5} {quant:>5} {size:>5}   failed: {exc}")
                        continue
                    # warmup() is what probes resize support, so ask after.
                    recognizer.warmup()
                    if size and not recognizer.image_size:
                        print(
                            f"{short:<30} {beams:>5} {quant:>5} {size:>5}   "
                            "skipped: this transformers cannot resize the encoder"
                        )
                        del recognizer, pipeline
                        continue
                    times, mean_cer, _ = measure(pipeline, samples)
                    median = statistics.median(times)
                    p90 = sorted(times)[max(0, int(len(times) * 0.9) - 1)]
                    cer_text = f"{mean_cer:.3f}" if mean_cer is not None else "   -- "
                    print(
                        f"{short:<30} {beams:>5} {quant:>5} "
                        f"{recognizer.image_size or 384:>5} "
                        f"{median:>7.2f}s {p90:>6.2f}s {cer_text:>6}"
                    )
                    del recognizer, pipeline

    print(
        "\nPick the fastest row above whose CER is not meaningfully worse than "
        "the fp32/384px baseline for the model you'd actually ship.\n"
        "\nMeasured on the reference Pi, 2026-09-03, over the FULL sample set "
        "(eval_backend,\nnot a limited sweep): trocr-small runs ~6.8x faster "
        "than trocr-base (4.21s -> 0.62s)\nfor natural-language CER 0.425 -> "
        "0.486 — a real trade, not a free win. --quantize\nwas "
        "neutral-to-slower once its ARM crash was fixed, and --image-size 224 "
        "cost real\naccuracy on both models.\n"
        "\nDon't assume any of that transfers to your hardware or handwriting. "
        "And don't draw\naccuracy conclusions from a small --limit: this "
        "dataset is sorted easiest-first, so a\nlimited sweep sees the easy "
        "samples. Use eval_backend over everything for accuracy;\nuse this "
        "sweep for speed."
    )


if __name__ == "__main__":
    main()
