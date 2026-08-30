"""One-time export of a TrOCR handwritten model to ONNX for the 'trocr' backend.

Usage:
    python -m scripts.export_trocr_onnx [--model NAME] [--out DIR]

Defaults to the small handwritten checkpoint, which is the best speed/accuracy
trade-off on a Raspberry Pi 5. Use ``microsoft/trocr-base-handwritten`` for
higher accuracy if you can tolerate several seconds per line.
"""

from __future__ import annotations

import argparse

DEFAULT_MODEL = "microsoft/trocr-small-handwritten"
DEFAULT_OUT = "models/trocr-small-handwritten-onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import TrOCRProcessor
    except ImportError:
        raise SystemExit(
            "Missing packages. Install them first:\n"
            "  pip install -r requirements-trocr.txt"
        )

    print(f"Exporting {args.model} -> {args.out} (this downloads the model once)…")
    model = ORTModelForVision2Seq.from_pretrained(args.model, export=True)
    model.save_pretrained(args.out)
    TrOCRProcessor.from_pretrained(args.model).save_pretrained(args.out)
    print("Done. Run the app with:  ./run.sh --backend trocr")


if __name__ == "__main__":
    main()
