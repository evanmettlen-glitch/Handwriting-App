"""Fine-tune TrOCR on samples collected with ``./run.sh --train``.

Run this on a machine with a CUDA GPU if you can — a few hundred samples still
finish in minutes. The Raspberry Pi only ever runs inference.

    python -m scripts.finetune_trocr --samples data/samples --out models/trocr-personal

Then export the result to ONNX for the app:

    python -m scripts.export_trocr_onnx --model models/trocr-personal \
        --out models/trocr-personal-onnx --quantize
    ./run.sh --model-dir models/trocr-personal-onnx

Use ``--dry-run`` first to check the collected data without loading PyTorch.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from handwriting_app.dataset import iter_samples

DEFAULT_BASE = "microsoft/trocr-small-handwritten"
RENDER_KW = dict(deslant=True, supersample=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--out", default="models/trocr-personal")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="Inspect the data and exit.")
    return p.parse_args()


def load_split(samples_dir: str, val_frac: float, seed: int):
    samples = list(iter_samples(samples_dir))
    if not samples:
        raise SystemExit(
            f"No samples in {samples_dir!r}. Collect some first with ./run.sh --train"
        )
    random.Random(seed).shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac)) if len(samples) > 10 else 0
    return samples[n_val:], samples[:n_val]


def dry_run(samples_dir: str) -> None:
    samples = list(iter_samples(samples_dir))
    print(f"{len(samples)} samples in {samples_dir}")
    labels = [s.label for s in samples]
    uniq = sorted(set(labels))
    print(f"{len(uniq)} unique labels")
    widths = []
    for s in samples[:200]:
        img = s.ink.render(stroke_width=s.stroke_width, **RENDER_KW)
        if img is not None:
            widths.append(img.size)
    if widths:
        ws = [w for w, _ in widths]
        hs = [h for _, h in widths]
        print(f"rendered size  w: {min(ws)}–{max(ws)}   h: {min(hs)}–{max(hs)}")
    print("first labels:", ", ".join(labels[:12]))
    dupes = len(labels) - len(uniq)
    if dupes:
        print(f"{dupes} repeated labels (fine — repeats help)")


def _augment(image, rng):
    """Small random affine so limited data generalizes better."""
    from PIL import Image

    angle = rng.uniform(-4, 4)
    scale = rng.uniform(0.9, 1.1)
    image = image.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=255)
    w, h = image.size
    image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return image


def build_dataset(samples, processor, augment, seed):
    import torch

    class InkDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = samples
            self.rng = random.Random(seed)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            sample = self.samples[i]
            image = sample.ink.render(stroke_width=sample.stroke_width, **RENDER_KW)
            if image is None:
                image = _blank()
            image = image.convert("RGB")
            if augment:
                image = _augment(image, self.rng)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
            token_ids = processor.tokenizer(
                sample.label, padding="max_length", max_length=32, truncation=True
            ).input_ids
            labels = [t if t != processor.tokenizer.pad_token_id else -100 for t in token_ids]
            return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

    return InkDataset()


def _blank():
    from PIL import Image

    return Image.new("L", (64, 64), 255)


def char_error_rate(pred: str, gold: str) -> float:
    # Levenshtein / len(gold)
    m, n = len(pred), len(gold)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (pred[i - 1] != gold[j - 1]),
            )
            prev = cur
    return dp[n] / max(1, n)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args.samples)
        return

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        raise SystemExit(
            "Training needs PyTorch + Transformers:\n"
            "  pip install -r requirements-train.txt"
        )

    train_samples, val_samples = load_split(args.samples, args.val_frac, args.seed)
    print(f"train: {len(train_samples)}   val: {len(val_samples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.base)
    model = VisionEncoderDecoderModel.from_pretrained(args.base).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    train_ds = build_dataset(train_samples, processor, args.augment, args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            running += loss.item()
        line = f"epoch {epoch:2d}  loss {running / max(1, len(train_dl)):.4f}"

        if val_samples:
            model.eval()
            total_cer, n = 0.0, 0
            with torch.no_grad():
                for sample in val_samples:
                    image = sample.ink.render(stroke_width=sample.stroke_width, **RENDER_KW)
                    if image is None:
                        continue
                    pv = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
                    ids = model.generate(pixel_values=pv, max_new_tokens=32)
                    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    total_cer += char_error_rate(text.strip(), sample.label)
                    n += 1
            line += f"   val CER {total_cer / max(1, n):.3f}"
        print(line)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"\nSaved to {out}. Next: export to ONNX (see the header of this file).")


if __name__ == "__main__":
    main()
