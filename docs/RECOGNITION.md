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

### How Apple does it (and what that implies)

Apple's Scribble does **not** fine-tune a model per user. Publicly, their
approach is:

1. **Online, stroke-based.** The model consumes the pen trajectory — a sequence
   of points over time — not a rasterized image. Stroke order and direction
   disambiguate cursive in a way pixels cannot.
2. **One general model trained on a large multi-writer corpus.** A model that has
   seen thousands of hands generalizes to a new one with zero user-specific
   training.
3. **Personalization lives in the language model / lexicon**, not the recognizer
   weights — contacts, vocabulary, and context bias the decoding.
4. Small enough to run on-device in real time.

The lesson for this project: *more training data from one user is the wrong
axis*. The productive axes are a better general model (A), input/output
adaptation around it (B'), and stroke-based input (C).

### B'. Calibration — personalization without training  *(implemented)*

The cheap, Apple-shaped version of personalization. `scripts/calibrate.py` makes
one forward pass over the collected samples and writes
`data/samples/calibration.json`:

- **best render settings** — grid-searches deslant / stroke width / padding
  against the user's own labels, keeps the lowest-CER combination
- **whole-word fixes** — words the recognizer misreads the *same way* at least
  twice become a substitution table
- reports baseline vs tuned CER so the gain is visible

Works from ~20 samples, takes minutes, no gradients. The app loads it
automatically (`--no-calibration` opts out). Combined with `lexicon.py`, this is
the whole personalization story for a small sample set.

### B. Personalized fine-tuning — only with enough data

**Measured on 40 samples: val CER went 0.52 → 0.80.** The model memorized the 36
training samples and got worse on the 4 held out. Fine-tuning TrOCR needs
~150–300+ samples before it beats the stock checkpoint; below that, use B'.

- **`./run.sh --train`** (`handwriting_app/training.py`) runs a guided
  enrollment: the ~40-prompt set in `enrollment.py`, ordered for full a-z / A-Z /
  0-9 coverage in under 5 minutes, with a progress bar, timer, time-left
  estimate, and live coverage readout. Each **Save & next** stores
  `NNNN_label.json` (raw strokes) + a `.png` preview under `data/samples/`.
  Resume is automatic. `--freeform` / `--prompts-file` use the open-ended list.
- **`scripts/finetune_trocr.py`** loads the samples, renders them through the
  *same* `Ink.render(deslant=True)` the live pipeline uses, applies light affine
  augmentation, and fine-tunes `trocr-small-handwritten` for a few epochs
  (reports train loss + val CER). `--dry-run` inspects the data without torch.
- `scripts/train_personal.sh [name]` does fine-tune + ONNX export in one step.
- The app then loads it automatically: `models.resolve_model_dir()` prefers
  `models/<user>-onnx` (with `--user`), then `models/trocr-personal-onnx`, then a
  generic model, then tesseract.
- **Zero-training personalization** (`lexicon.py`): every word from the collected
  sample labels is added to the SymSpell dictionary as a known term, so the
  corrector stops turning the user's names/jargon into dictionary words. Active
  by default whenever `data/samples/` has anything in it.

### C. Online stroke-based model — what Apple actually does

Feed the network the pen trajectory: sequences of `(dx, dy, pen_up, pen_down)`
per timestep. Classic architecture: bidirectional LSTM + CTC loss (Graves 2009),
or a small transformer encoder + CTC. Train once on **IAM-OnDB** (online
handwriting, ~13k labelled lines) — a general model, not a per-user one.

- Model is tiny (~1–5M params) and fast on the Pi.
- ~85–90% character accuracy achievable; add the same SymSpell / KenLM pass on top.
- We already record `(x, y)` per point; add timestamps in `Stroke.add` and the
  input features are ready. The collected samples are stored as strokes, so they
  are already usable as training/eval data for this.
- Effort: real training pipeline + the dataset. Weeks, not days.

Reference points: `SimpleHTR` (offline, word-level CNN-RNN-CTC, ~5M params, a
good lightweight fallback), and published IAM-OnDB BiLSTM-CTC implementations.

## Roadmap

- [x] **Phase 0** — tesseract baseline + image preprocessing.
- [x] **Phase 1** — pen-lift word segmentation, slant normalization, SymSpell
  correction, TrOCR as the default when its model is present.
- [x] **Phase 2** — `--train` data-collection mode + `finetune_trocr.py`.
  Conclusion: fine-tuning needs far more data than a 5-minute enrollment gives.
- [x] **Phase 2b** — `calibrate.py`: no-training personalization (render search +
  word fixes) plus the personal lexicon. This is the right layer for ~40 samples.
- [ ] **Phase 3** — online BiLSTM-CTC model trained on IAM-OnDB; add timestamps
  to strokes; make it selectable as `--backend online`. **This is the real fix**
  for sloppy writing and cursive, and matches how Apple does it.
- [ ] **Phase 4** — KenLM/n-gram decoding for context-aware correction.

## Datasets

- **IAM Handwriting DB** — offline lines/words, the TrOCR training set. Free for
  non-commercial research; register at the FKI website.
- **IAM-OnDB** — online (stroke) handwriting, for phase 3.
- **CVL**, **RIMES** (French), **Bentham** — additional offline sets.
- Your own samples from training mode — worth more than all of the above for a
  single-user device.
