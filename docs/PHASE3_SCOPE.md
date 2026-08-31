# Phase 3 scope — online stroke-based recognition

**Goal:** replace image-OCR with a model that reads the pen *trajectory*, so
sloppy writing and cursive work. This is what Apple does. Target: beat the TrOCR
baseline on real touchscreen input, at >10× the speed.

**Honest effort:** ~12–18 focused days, 3–5 weeks wall-clock part-time.
Two hard dependencies (dataset access, GPU time) and one significant technical
risk (domain gap) that could sink it — there's a kill-gate at stage 4 to find
out cheaply.

---

## Why this is the right move

| | Current (TrOCR, image) | Phase 3 (BiLSTM-CTC, strokes) |
|---|---|---|
| Input | flattened bitmap | pen trajectory over time |
| Cursive | weak — letters merge visually | strong — stroke order disambiguates |
| Model size | 62M (small) / 334M (base) | **~1–2M** |
| Latency on Pi 5 | 1–5 s / line | **<100 ms / line** |
| Per-user fine-tune | needs 150–300+ samples | viable at ~100 (small model) |

The latency alone changes the product: sub-100 ms means recognition can run
*while you write* instead of after a pause.

---

## Architecture

```
Ink (strokes)
   │
   ├─ resample to uniform arc-length spacing      ← kills writing-speed variance
   ├─ normalize (center, scale by x-height)
   └─ per-point features → T × 8 matrix
   │
BiLSTM ×3, hidden 128/direction  (~1.2M params)
   │
Linear → T × 81 logits            (80 chars + CTC blank)
   │
CTC decode  (greedy → beam search + personal lexicon)
   │
text
```

**Features per resampled point (8 dims):**
`Δx, Δy, |Δ|, cos θ, sin θ, curvature, pen-up flag, normalized y`

Deliberately **geometric, not temporal** — arc-length resampling removes timing
on purpose, which is standard practice (Graves 2009). Two consequences:

1. **The 40 samples already collected stay valid.** They have no timestamps and
   don't need any.
2. Timestamps should still be recorded going forward (one-line change, cheap
   insurance) in case later features want velocity.

---

## Stages

### Stage 0 — evaluation harness *(1–2 days)* — **do this first**

Nothing else is measurable without it.

- `scripts/eval_backend.py`: run any backend over a sample folder, report CER,
  word accuracy, and per-character confusion.
- Establishes the number to beat: **current TrOCR-base CER on your 40 samples.**
- Reuses `textalign.cer()` and `char_confusions()`, which already exist.

Deliverable: one command that prints a CER for any backend. Everything after
this is judged against it.

### Stage 1 — stroke features *(1–2 days)*

- `handwriting_app/features.py`: `ink_to_features(ink) -> np.ndarray (T, 8)`
- Arc-length resampling, normalization, the 8 features above.
- Add timestamps to `Stroke.add()` (unused by v1, future-proofing).
- Unit tests: scale/translation invariance, resampling stability, a stroke drawn
  fast vs slow producing near-identical features.

### Stage 2 — IAM-OnDB ingestion *(1–2 days, + registration wait)*

- **Register at the FKI (Univ. Bern) site now** — free for non-commercial
  research, but approval is not instant. This gates stages 3–4, so start it
  before anything else.
- ~13k labelled lines, 221 writers, XML stroke format.
- `scripts/prepare_iam_ondb.py`: parse XML → the same `Ink` JSON format the app
  already writes, so one code path feeds both.
- Split by *writer*, not by line, or validation scores are inflated.

⚠️ **Licence:** IAM-OnDB is non-commercial research use only. Fine for this
project; would need revisiting if it ever ships commercially.

### Stage 3 — model and training loop *(2–3 days)*

- `handwriting_app/online_model.py`: the BiLSTM-CTC module (~1.2M params).
- `scripts/train_online.py`: `torch.nn.CTCLoss`, Adam, cosine schedule.
- Augmentation is critical given the domain gap — random affine on the *point
  sequence*, stroke jitter, point dropout (simulates a low-rate touchscreen),
  slant and aspect variation.
- Checkpoint on best validation CER.

### Stage 4 — train, and the **kill-gate** *(3–10 days wall-clock)*

Train on IAM-OnDB, then measure on **your** 40 samples with the stage-0 harness.

- **Expected:** IAM-OnDB val CER 0.08–0.15 (published results land here).
- **The real question:** CER on touchscreen samples. If the domain gap is bad
  this could be 0.4+ even with a healthy validation score.

> **Gate:** if CER on real touchscreen input isn't clearly better than the TrOCR
> baseline after augmentation and a fine-tune pass on collected samples,
> **stop and keep TrOCR.** Don't sink stages 5–6 into a model that lost.

Fallback if the gap is the problem: fine-tune the trained model on 100–300
collected samples. At 1–2M params this actually works — unlike the failed 62M
TrOCR fine-tune (which went CER 0.52 → 0.80 on 40 samples).

### Stage 5 — decoding with the personal lexicon *(2–3 days)*

Greedy CTC first (trivial, ships in stage 4). Then the part that matters:

- Beam search with a lexicon/LM constraint — `pyctcdecode` + KenLM, or
  `torchaudio.models.decoder.ctc_decoder`.
- **This is where `lexicon.py` finally pays off properly.** Instead of correcting
  after the fact, your vocabulary biases decoding *during* the search — which is
  precisely how Apple personalizes. Typically worth 20–40% relative CER.

### Stage 6 — Pi integration *(2–3 days)*

- Export to ONNX; verify LSTM op support in ONNX Runtime (it's fine, but check).
- `OnlineRecognizer` selected via `--backend online`.
- ⚠️ **Interface change:** the current `Recognizer` ABC is `recognize(image)`.
  The online model needs strokes. Cleanest fix: an `InkRecognizer` protocol
  alongside it, with `RecognitionPipeline` dispatching on which one it got.
  Roughly a half-day of refactor — the pipeline keeps both paths so TrOCR
  survives as a fallback.
- Benchmark on the Pi; confirm the latency claim.

---

## Risks, highest first

| Risk | Impact | Mitigation |
|---|---|---|
| **Domain gap** — IAM-OnDB is whiteboard/eBeam pen at ~100 Hz; you have finger-on-touchscreen with Tk-coalesced, irregular, sparse points | Model trains well, works badly for you | Arc-length resampling (removes rate mismatch), point-dropout augmentation, fine-tune on collected samples. **This is what stage 4's gate tests.** |
| **Dataset access** — FKI registration | Blocks stages 3–4 entirely | Register on day 1, before writing any code |
| **GPU time** — CPU training is days per run | Slow iteration kills the project | Colab free tier, or rent an hour of a GPU. The Pi does inference only |
| Interface refactor breaking the image path | Regression | Keep both protocols; TrOCR stays the fallback |
| Punctuation/digits underrepresented in IAM-OnDB | Bad on `$19.95`-type input | Weight the loss, or synthesize; your enrollment set already covers these for eval |

---

## Decisions I need from you

1. **Where does training compute come from?** Colab free tier is enough for a
   1M-param model (hours, not days). Do you have a machine with a GPU?
2. **Line-level or word-level?** Word-level is easier to train and matches the
   existing pen-lift segmentation. Recommend word-level for v1.
3. **Register for IAM-OnDB now?** It's the long-lead item and costs nothing to
   start.

## Suggested first move

Stage 0 alone — the eval harness. It's 1–2 days, it's useful immediately
regardless of whether phase 3 ever happens (it tells you whether calibration and
the base model are helping), and it produces the baseline number that every
later decision depends on.

I'd build stage 0 next, then start the IAM-OnDB registration while it runs.
