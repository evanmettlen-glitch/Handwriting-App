# Recognition: research notes and roadmap

Goal: read sloppy printing **and** cursive, use pen-lift information, and emit
real English words — all offline on a Raspberry Pi 5.

## Why the first version is weak

v1 flattens every stroke onto one bitmap and runs `tesseract`. That throws away
the two things that make handwriting legible to a machine:

1. **Stroke / temporal information** — the order and direction a person draws in.
   Online (stroke-based) recognizers beat image-based ones on cursive precisely
   because of this.
2. **Pen lifts** — we already capture each stroke separately in `Ink.strokes`,
   then destroy that structure by merging into one image.

And `tesseract` is an OCR engine trained on scanned print; it has no useful prior
over cursive letterforms and no real language model.

## The three approaches

| Approach | Handles cursive | Uses pen lifts | Offline on Pi 5 | Effort |
|---|---|---|---|---|
| **A. Image model (TrOCR)** | yes (trained on IAM) | indirectly, via word segmentation | yes, ~1–5 s/line | low — mostly wiring |
| **B. Personalized fine-tune** | yes, very well for *one* hand | as A | yes | medium — needs a data-collection UI + a training run |
| **C. Online stroke model** | yes, best | natively | yes (small model) | high — train on IAM-OnDB |

### A. Image model — TrOCR handwritten

`microsoft/trocr-*-handwritten` is a ViT encoder + RoBERTa-style decoder trained
on the IAM handwriting database (natural cursive). The decoder *is* a language
model, so output tends to be real words.

- `trocr-small-handwritten` (62M) — ~1–2 s/line on Pi 5 CPU, decent.
- `trocr-base-handwritten` (334M) — ~3–5 s/line, clearly better. **Recommended.**
- Quantize to int8 with ONNX Runtime to roughly halve latency and memory.

Improvements layered on top (implemented in phase 1):

- **Word segmentation from pen lifts** (`segmentation.py`): group strokes into
  words by horizontal gaps, recognize each word image separately, then join.
  Short images are far more reliable for every backend.
- **Slant normalization** (`ink.render(deslant=True)`): decorrelate x from y
  across all ink points to straighten a consistent rightward lean. Done on
  coordinates, before rasterization, so there are no resampling artifacts.
- **Dictionary correction** (`postprocess.py`): SymSpell against an English
  frequency dictionary maps near-misses to real words; optional compound mode
  also fixes wrong / missing spaces.

### B. Personalized fine-tuning — the real accuracy unlock

Generic cursive recognition is hard. Recognizing *your* cursive after ~150–300
labelled samples is very achievable.

Plan:

1. **Training mode** in the app: it shows a prompt word/phrase, you write it, it
   stores `(ink.json, label)` under `data/samples/`.
2. Render samples the same way the live pipeline does.
3. Fine-tune `trocr-small-handwritten` for a few epochs (a laptop GPU or a free
   Colab does this in minutes; the Pi only runs inference).
4. Drop the resulting weights in `models/` and point `--model-dir` at them.

This is the highest value-per-hour path after phase 1.

### C. Online stroke-based model — best cursive, most work

Feed the network the actual pen trajectory: sequences of
`(dx, dy, pen_up, pen_down)` per timestep. Classic architecture: bidirectional
LSTM + CTC loss (Graves 2009), or a small transformer encoder + CTC. Train on
**IAM-OnDB** (online handwriting, ~13k labelled lines).

- Model is tiny (~1–5M params) and fast on the Pi.
- ~85–90% character accuracy achievable; add the same SymSpell / KenLM pass on top.
- We already record `(x, y)` per point; add timestamps in `Stroke.add` and the
  input features are ready.
- Effort: real training pipeline + the dataset. Weeks, not days.

Reference points: `SimpleHTR` (offline, word-level CNN-RNN-CTC, ~5M params, a
good lightweight fallback), and published IAM-OnDB BiLSTM-CTC implementations.

## Roadmap

- [x] **Phase 0** — tesseract baseline + image preprocessing.
- [~] **Phase 1** — pen-lift word segmentation, slant normalization, SymSpell
  correction, TrOCR-base as the default when its model is present. *(this change)*
- [ ] **Phase 2** — training mode + fine-tuning scripts for a personal model.
- [ ] **Phase 3** — online BiLSTM-CTC model trained on IAM-OnDB; add timestamps
  to strokes; make it selectable as `--backend online`.
- [ ] **Phase 4** — KenLM/n-gram decoding for context-aware correction.

## Datasets

- **IAM Handwriting DB** — offline lines/words, the TrOCR training set. Free for
  non-commercial research; register at the FKI website.
- **IAM-OnDB** — online (stroke) handwriting, for phase 3.
- **CVL**, **RIMES** (French), **Bentham** — additional offline sets.
- Your own samples from training mode — worth more than all of the above for a
  single-user device.
