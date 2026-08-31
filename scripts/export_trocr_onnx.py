"""One-time export of a TrOCR handwritten model to ONNX for the 'trocr' backend.

Usage:
    python -m scripts.export_trocr_onnx [--model NAME] [--out DIR] [--quantize]

Defaults to the small handwritten checkpoint (safe on low-RAM Pis). For clearly
better accuracy on a Pi 5 with >=4 GB RAM, use the base model:

    python -m scripts.export_trocr_onnx \
        --model microsoft/trocr-base-handwritten \
        --out models/trocr-base-handwritten-onnx --quantize
"""

from __future__ import annotations

import argparse

DEFAULT_MODEL = "microsoft/trocr-small-handwritten"
DEFAULT_OUT = "models/trocr-small-handwritten-onnx"
_COMPONENTS = ("encoder_model", "decoder_model", "decoder_with_past_model")


def _quantize(out_dir: str) -> None:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
    for component in _COMPONENTS:
        quantizer = ORTQuantizer.from_pretrained(out_dir, file_name=f"{component}.onnx")
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
    print("Wrote *_quantized.onnx; the backend picks these up automatically.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also produce int8 models (faster + smaller, slight accuracy cost).",
    )
    args = parser.parse_args()

    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import TrOCRProcessor
    except ImportError:
        raise SystemExit(
            "Missing packages. Install them first:\n"
            "  pip install -r requirements-trocr.txt"
        )

    print(f"Exporting {args.model} -> {args.out} (downloads the model once)…")
    model = ORTModelForVision2Seq.from_pretrained(args.model, export=True)
    model.save_pretrained(args.out)
    TrOCRProcessor.from_pretrained(args.model, use_fast=False).save_pretrained(args.out)

    if args.quantize:
        print("Quantizing to int8…")
        try:
            _quantize(args.out)
        except Exception as exc:  # noqa: BLE001
            print(f"Quantization failed ({exc}); the float model is still usable.")

    print(f"Done. Run:  ./run.sh --backend trocr --model-dir {args.out}")


if __name__ == "__main__":
    main()
