# Handoff — Handwriting → Text

Touchscreen handwriting recognition for a Raspberry Pi 5. Write on the screen
with a finger or stylus, get editable text. Fully offline.

**Repo:** github.com/evanmettlen-glitch/Handwriting-App (public)
**Hardware:** Raspberry Pi 5 "HomeHubSSD", HDMI display + USB touch panel
**State as of 2026-08-31:** working end to end; accuracy is the open problem.
20 commits, 59 tests passing.

---

## Start here

```bash
ssh evmett@HomeHubSSD
cd ~/HandWritingApp
./run.sh                 # or ./run.sh --fullscreen for kiosk
```

First recognition takes ~30 s (model load), then ~3–8 s per line. The status
line at the bottom tells you exactly what is active:

```
Ready · backend: trocr-torch:trocr-base-handwritten
        (personal lexicon: 78 words; calibrated on 43 samples; whole line)
```

Fresh install on a new Pi:

```bash
git clone https://github.com/evanmettlen-glitch/Handwriting-App.git ~/HandWritingApp
cd ~/HandWritingApp && chmod +x *.sh scripts/*.sh
./install.sh                                    # system deps + venv + Pillow
./.venv/bin/pip install -r requirements-trocr.txt   # the neural backend (~1 GB)
./run.sh
```

The TrOCR model downloads itself on first run. No export step, no `optimum`.

---

## The two-machine workflow

This trips people up constantly. **The Pi never pushes.**

| | Windows PC | Raspberry Pi |
|---|---|---|
| Path | `~\OneDrive\Desktop\HandWritingApp` | `~/HandWritingApp` |
| Role | edit, commit, **push** | **pull**, run, collect data |
| Shell | PowerShell 5.1 — **no `&&`**, use `;` | bash |

Run Pi commands from the PC without switching windows — everything inside the
quotes is bash, so `&&` works there:

```bash
ssh evmett@HomeHubSSD "cd ~/HandWritingApp && git pull && ./run.sh --help"
```

Running the GUI over SSH needs a display hint:
`DISPLAY=:0 ./run.sh` (X11), or
`WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/1000 ./run.sh` (Wayland).

---

## How it works

```
finger/stylus → USB touch panel → Tk pointer events
      │
   InkCanvas          each pen-down..up is a Stroke of (x, y) points
      │
   segment_words      OFF for TrOCR (line model), ON for tesseract
      │
   Ink.render()       Catmull-Rom spline → deslant → B/W image
      │
   Recognizer         TrOCR (torch) or tesseract, on a worker thread
      │
   join_split_letters "a n d" → "and"
      │
   SpellCorrector     SymSpell + your personal vocabulary
      │
   Text box           edit · copy
```

### Module map

| File | Does |
|---|---|
| `app.py` | Tk UI, worker thread, result queue |
| `canvas_widget.py` | stroke capture, undo, redraw |
| `ink.py` | Stroke/Ink model, spline smoothing, deslant, rasterize |
| `segmentation.py` | split ink into words by pen-lift gaps |
| `pipeline.py` | segment → recognize → join → correct. `resolve_segment()` |
| `postprocess.py` | SymSpell correction, `join_split_letters()` |
| `lexicon.py` | personal word list mined from sample labels |
| `calibration.py` | `calibration.json` — render settings + word fixes |
| `models.py` | which model to load (personal → generic → HF default) |
| `training.py` + `enrollment.py` | `--train` enrollment UI |
| `textalign.py` | CER, character confusions |
| `recognizer/` | `base.py` ABC, tesseract, TrOCR torch, TrOCR ONNX |

### Scripts

```bash
python -m scripts.inspect_ink --sweep    # capture + segmentation diagnostics (fast)
python -m scripts.eval_backend           # CER / accuracy (the number to beat)
python -m scripts.bench_latency          # where the seconds go; speed vs CER sweep
python -m scripts.calibrate              # tune render settings, no training (~45 min)
python -m scripts.finetune_trocr         # fine-tune (needs 150-300+ samples)
./scripts/train_personal.sh [name]       # fine-tune + ONNX export in one go
```

---

## The accuracy investigation — read this before changing anything

This is the most valuable part of the handoff. Several plausible theories were
tested and **disproved**; don't re-run them.

### Confirmed root cause: letter spacing

Real predictions from the Pi (TrOCR-base, 43 enrolled samples):

```
'the'             -> 'the'          perfect
'you'             -> 'you'          perfect
'and'             -> 'a n d'        CER 0.67
'the quick brown' -> 'the quick-'
'fox jumps over'  -> 'fox , in the'
```

Short single words are **perfect**. That clears the model, the rendering, the
capture, and the smoothing. TrOCR is trained on *connected* handwriting, and
this user prints with gaps wide enough that each letter reads as its own word.

Mitigation shipped: `join_split_letters()` finds runs of ≥2 single-letter tokens
and greedily consumes the longest prefix the dictionary confirms.
`a n d`→`and`, `w i t h`→`with`, `I a m h e r e`→`I am here`, while `x q z`,
`a`, and ordinary text pass through untouched.

**The bigger lever is behavioral: write letters closer together.** It also makes
word gaps unambiguous, which fixes segmentation as a side effect.

### Ruled out — do not revisit

| Theory | Evidence against |
|---|---|
| **Sparse capture / Tk event coalescing** | 407 points/sample, 4.5 px median gap over 604 px of writing. Only 7/43 marginal. Density is fine. |
| **Bad render settings** | Grid search moved CER only 0.50–0.60. Not the bottleneck. |
| **Systematic letterform confusion** | Calibration found **zero** consistent word fixes. Errors are random, not stylistic. |
| **Word segmentation threshold** | Sweep plateaus at 21/43 wrong for *every* ratio 0.5–2.0. No threshold exists, because letter gaps ≈ word gaps. |
| **Fine-tuning TrOCR on the enrollment set** | Measured: val CER **0.52 → 0.80** on 40 samples. It memorizes and generalizes worse. Needs 150–300+. |
| **SymSpell compound mode for split letters** | Measured: `w i t h` → `a it a`, `a n d` → `an a`. Makes it worse. |

### Latency (the open problem as of 2026-08-31)

Accuracy is now acceptable; **time to result is the complaint**. The whole cost
is TrOCR-base on a CPU — render, segmentation, and spell correction are
sub-millisecond. Measure with `python -m scripts.bench_latency`, which reports
throttle state, a per-stage breakdown, and a speed-vs-CER sweep.

Shipped, no accuracy risk:

- **Startup warm-up.** The first inference used to cost ~30 s of lazy init on
  the user's first real line. `Recognizer.warmup()` now runs a throwaway pass
  during load, before the Recognize button enables.
- **A counting-up status line** (`Recognizing…  2.4s`) and elapsed time on the
  result, so a slow run reads as slow rather than hung.

Knobs, biggest lever first — all measurable with the bench:

| Flag | Effect | Risk |
|---|---|---|
| `--model-dir microsoft/trocr-small-handwritten` | ~5x less compute | real accuracy risk |
| `--quantize` | ~2x faster (dynamic int8) | some accuracy |
| `--beams 1` | ~4x less decode — **now the default** | negligible |

`--beams 1` is the one free win: the `microsoft/trocr-*` checkpoints ship
`num_beams=4` in their generation config, so every recognition was running beam
search four ways over 577 encoder patches by default.

### Segmentation is off for TrOCR on purpose

TrOCR was trained on IAM *text lines* and its decoder uses cross-word context, so
a whole-line image is what it wants. Word segmentation was a hack for tesseract,
which is weak on multi-word images. `pipeline.resolve_segment()` picks
automatically; `--segment` / `--no-segment` override.

### How Apple does it

Apple's Scribble does **not** fine-tune per user. It uses an *online,
stroke-based* model (pen trajectory, not pixels) trained once on a large
multi-writer corpus, and personalizes only the lexicon/language model. The
implication drove the design here: more training data from one user is the wrong
axis. Hence `calibrate.py` and `lexicon.py` instead of per-user fine-tuning.

---

## Personalization (all automatic, no flags)

1. **Model** — `models.resolve_model_dir()` prefers `models/<user>-onnx` (with
   `--user`), then `models/trocr-personal*`, then a generic export, then
   downloads `microsoft/trocr-base-handwritten`.
2. **Personal lexicon** — every word from collected sample labels is boosted in
   SymSpell, so it stops "correcting" names and jargon. Verified: `Priya` stays
   `Priya` instead of becoming `Oriya`, while `teh`→`the` still works.
3. **Calibration** — `data/samples/calibration.json` carries the best render
   settings, word-gap ratio, and word fixes.

Collect data with `./run.sh --train`: a guided ~40-prompt enrollment with a
progress bar, timer, and live a-z/A-Z/0-9 coverage. Designed for under 5 minutes.
`--user NAME` scopes to `data/samples/NAME/`.

---

## Gotchas that cost real time

- **`optimum` is incompatible with torch 2.13** (`_attention_scale` removed from
  `torch.onnx.symbolic_opset14`). ONNX export is optional and currently
  unusable on this Pi. The torch path works fine — don't reintroduce the
  dependency without checking.
- **`transformers` v5 breaks TrOCR's tokenizer.** All `TrOCRProcessor`
  loads use `use_fast=False`.
- **`calibration.json` lives inside `data/samples/`.** `dataset.sample_paths()`
  skips known sidecars; anything else added there must be added to
  `NON_SAMPLE_FILES` or it will be parsed as a handwriting sample.
- **Enrollment resumes by matching prompt labels**, not by counting files. An
  earlier version counted files and jumped straight to "All prompts done".
- **Rote prompts skew accuracy metrics.** `abcdefghijklm`, `0123456789` etc.
  exist for character coverage; a language-model decoder mangles them however
  neatly they were written. `eval_backend` reports them separately — always
  judge on the **natural language** line. One rote sample can drag an aggregate
  from CER 0.05 to 0.75.

---

## What to do next

**Immediate (minutes):**
1. `python -m scripts.bench_latency` — pick the fastest row whose CER matches
   the `beams=1 / int8=off` baseline, then run the app with those flags.
2. Re-measure accuracy with letter-joining active:
   `python -m scripts.eval_backend --limit 12`
3. If improved, re-run `python -m scripts.calibrate` (~45 min) to re-tune with
   the new segmentation default and bake it into `calibration.json`.

**If accuracy is still short:**
- Collect 150–300 samples (`./run.sh --train --freeform`, repeat) and *then*
  fine-tuning becomes viable — `./scripts/train_personal.sh`.
- Or go to phase 3.

**Phase 3 — online stroke-based recognition.** Fully scoped in
[PHASE3_SCOPE.md](PHASE3_SCOPE.md): BiLSTM+CTC over the pen trajectory, ~1–2M
params, trained on IAM-OnDB. Would take latency from 3–8 s to <100 ms and is the
principled fix for cursive. ~12–18 focused days, 3–5 weeks part-time. Two hard
dependencies (FKI dataset registration — **start early, it gates everything**;
GPU time) and one real risk (domain gap between whiteboard-captured IAM-OnDB and
finger-on-touchscreen input). There is a deliberate kill-gate at stage 4 so a
losing model gets abandoned before the expensive integration work.

Stage 0 of that plan (the eval harness) is already built and in use.

**Open decisions:** where training compute comes from; word-level vs line-level
for v1 (recommend word-level); whether to register for IAM-OnDB now.

---

## Reference

- [README.md](../README.md) — install, flags, kiosk autostart, touch calibration
- [docs/RECOGNITION.md](RECOGNITION.md) — approaches, tradeoffs, roadmap
- [docs/PHASE3_SCOPE.md](PHASE3_SCOPE.md) — the stroke-model plan
- Tests: `python -m pytest` (59, all passing)
