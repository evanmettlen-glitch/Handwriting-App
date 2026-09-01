# Handoff — Handwriting → Text

Touchscreen handwriting recognition for a Raspberry Pi 5. Write on the screen
with a finger or stylus, get editable text. Fully offline.

**Repo:** github.com/evanmettlen-glitch/Handwriting-App (public)
**Hardware:** Raspberry Pi 5 "HomeHubSSD", HDMI display + USB touch panel
**State as of 2026-08-31:** working end to end; accuracy is the open problem.
21 commits, 121 tests passing.

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
   clean_ink          cut drags and no-lift slides, drop stray taps
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
| `cleanup.py` | remove drags / no-lift slides / stray taps. `clean_ink()` |
| `postprocess.py` | SymSpell correction, `join_split_letters()`, `complete()` |
| `ink.py` | Stroke/Ink model, spline smoothing, deslant, rasterize |
| `segmentation.py` | split ink into words by pen-lift gaps |
| `pipeline.py` | segment → recognize → join → correct. `resolve_segment()` |
| `lexicon.py` | personal word list mined from sample labels |
| `calibration.py` | `calibration.json` — render settings + word fixes |
| `models.py` | which model to load (personal → generic → HF default) |
| `training.py` + `enrollment.py` | `--train` enrollment UI |
| `textalign.py` | CER, character confusions |
| `recognizer/` | `base.py` ABC, tesseract, TrOCR torch, TrOCR ONNX |

### Scripts

```bash
python -m scripts.inspect_ink --sweep    # capture + segmentation diagnostics (fast)
python -m scripts.inspect_cleanup       # what ink cleanup removes, and whether it helped
python -m scripts.eval_backend           # CER / accuracy (the number to beat)
python -m scripts.bench_latency          # where the seconds go; speed vs CER sweep
python -m scripts.calibrate              # tune render settings, no training (~45 min)
                                         #   tunes on cleaned ink; --no-cleanup opts out
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
| `--image-size 224` | ~3x less encoder work (577 patches -> 197) | needs measuring |
| `--beams 1` | ~4x less decode — **now the default** | negligible |

`--beams 1` is the one free win: the `microsoft/trocr-*` checkpoints ship
`num_beams=4` in their generation config, so every recognition was running beam
search four ways over 577 encoder patches by default.

`--image-size` is the newest knob and the least understood. The vision encoder's
cost is fixed per image and set entirely by the patch count: 384x384 at patch 16
is 577 patches, 224x224 is 197. Running below the trained size means
interpolating the ViT position embeddings, which recent `transformers` supports
and older ones do not — `TrocrTorchRecognizer` probes once at load with a
throwaway inference and falls back to 384, saying `resize unsupported` in the
status line, rather than failing on a real line. **Whether accuracy survives on
handwriting is unmeasured.** The bench sweeps it:

```bash
python -m scripts.bench_latency --image-sizes 0 256 224 --quantize off on
```

Also worth remembering that ~1.8 s of the wait is not the model: `auto_delay_ms`
is how long the app waits for writing to stop before it starts.

### Streaming the decode, and guessing the word — added 2026-08-31

None of the knobs above make the wait shorter than one encoder pass plus one
token at a time, so the wait is also attacked from the other side: make it
legible instead of dead. The decoder emits a token at a time, so the recognizer
now hands each one back through an `on_partial` callback and the status line
fills in as it goes:

```
Recognizing…  2.4s   ▸ the quick brow(n)
```

The parenthesised half completes the word being decoded from the dictionary and
the personal lexicon — `SpellCorrector.complete()`, preferring your own
vocabulary outright, because the English frequency counts run to billions and a
common word would otherwise always outrank a name you actually write.
`pipeline.predict()` holds the display policy: no guess past a tail that is
already a real word, since turning every `the` into `the(y)` is noise.

Three things to hold onto:

- **It is display only.** The committed text is always what the model produced.
  A guess never reaches the text box, and `predict()` is called from the UI
  thread, not from the decode loop.
- **It costs nothing on the recognition path.** The tokens were being generated
  anyway; the completion scan runs between decoder steps on the idle UI thread.
- **It does not make anything faster** — it makes the wait readable, and lets a
  wrong reading be spotted before it finishes. The seconds themselves only come
  off with the model, int8, and patch-count levers above.

`on_partial` is passed to `recognize()` only when a caller asks for partials, so
a backend that cannot stream never sees the argument. Tesseract and the ONNX
path both return whole lines and simply never call it. `--no-predict` drops the
guess; streaming itself is unconditional.

Two implementation notes that are easy to undo by accident:

- **`warmup()` is also the probe.** Streaming and `--image-size` are both
  version-dependent `generate` kwargs (beam search rules streaming out too), so
  they are settled by a throwaway inference. That inference *is* the warm-up —
  probing separately added a whole encoder pass, several seconds, to Pi startup.
  Consequently `recognizer.name` and `.streaming` only read true after
  `warmup()`, and `name` is a property rather than an attribute.
- **`complete()` keeps its own two-letter prefix index**, built lazily. SymSpell
  indexes by edit distance, so a prefix lookup would otherwise scan all ~80k
  words — once per generated token, on the UI thread.

**The real answer is still phase 3.** A stroke model at <100 ms would let
recognition run *while* you write rather than after a pause, which is what makes
prediction genuinely useful rather than cosmetic.

### Segmentation is off for TrOCR on purpose

TrOCR was trained on IAM *text lines* and its decoder uses cross-word context, so
a whole-line image is what it wants. Word segmentation was a hack for tesseract,
which is weak on multi-word images. `pipeline.resolve_segment()` picks
automatically; `--segment` / `--no-segment` override.

### Ink cleanup — sloppy input, added 2026-08-31

The enrollment set is tidier than real use: every sample in it was written
deliberately, one word at a time, with the pen lifted between words. Real use
brings three things it does not contain, and all three degrade the image handed
to the model rather than the model itself.

| Symptom | What lands in the ink |
|---|---|
| Finger or sleeve brushes the pad | a long flat line straight through the writing |
| Pen never lifts between words | the slide from one word to the next, drawn |
| Knuckle or palm taps the glass | a speck off to the side that inflates the bounding box, shrinking the real writing when the image is scaled to the model's input height |

`handwriting_app/cleanup.py` removes all three with one detector, because
geometrically they are one thing: a **traverse** — a long horizontal move
carrying no vertical information. No letter does that. A cursive ligature is
short (0.15–0.25× the line height) and rises and falls; a traverse runs a line
height or more and stays inside a narrow band. Cutting it out is also what
restores the missing pen lift, so `segment_words()` sees word boundaries again
in writing that was never lifted — one fix, two symptoms.

Three things it is worth knowing before touching the thresholds:

- **Per-step flatness does not work, cumulative rise does.** The obvious test —
  is this step nearly horizontal — fails at real capture density. A step along
  the leg of a letter is ~4.5 px long, and a pixel or two of touch jitter makes
  it look as flat as a drag. What separates them is the *total* vertical extent
  over the whole run.
- **Straightness in 2-D was the wrong second test** and got replaced by
  horizontal directness (`|net dx| / total |dx|`). Straightness double-counts
  the vertical wander the rise limit already caps, and on a short traverse the
  curl of the letters at either end sinks it. Directness still rejects what
  straightness was there for — a scribbled-out word stays just as flat.
- **Speed earns the shorter threshold.** A traverse must run a full line height
  to be cut, *unless* it is both fast (a stride several times the writing's
  typical 4.5 px — Tk's coalescing turns speed into sparse points) and interior
  (real writing on both sides of it). A dash or a flourish is its own stroke; a
  slide between two words is not. Neither signal alone is trusted.

Cleanup is a **recognition-time** step. `save_sample()` still stores raw
strokes, so the collected samples stay the source of truth and a threshold
change applies retroactively to everything already recorded.

**Accepted false positive:** a deliberate long horizontal rule — an underline or
a strikethrough — is a traverse by every test here and gets removed. That seems
like the right trade for a handwriting-to-text pad.

**What is not verified yet.** The thresholds were chosen and tested on synthetic
ink at Pi capture scale (~180 px writing height, 4.5 px steps). Before trusting
them, run them over real samples on the Pi:

```bash
python -m scripts.inspect_cleanup --dump-png out/
```

The 43 existing samples were all written *with* pen lifts, so they only
exercise the safety half — cleanup should report ~0% of ink removed on them, and
anything else is a bug. To exercise the other half, collect a handful written
without lifting the pen and a couple with a deliberate finger drag
(`./run.sh --train --freeform`), then re-run. The metric that matters is word
counts: cutting a slide should make `segment_words` agree with the label more
often. The metric to worry about is ink removed on a sample whose word count did
*not* improve.

**Where raw and cleaned ink still diverge.** Cleanup runs at recognition time,
so anything that renders ink for another purpose has to decide for itself:

| Path | Ink it uses | Why |
|---|---|---|
| the app, `eval_backend`, `bench_latency` | cleaned | they all go through `pipeline.run()` |
| `calibrate.py` | cleaned | tuning against an image the model never sees is worthless — `--no-cleanup` opts out |
| `dataset.save_sample` | **raw** | samples are the source of truth; a threshold change has to apply retroactively |
| `inspect_ink.py` | **raw** | its whole subject is capture quality, before anything touches it |
| `finetune_trocr.py` | **raw** | *unresolved* — see below |

The fine-tuning one is a real open question. If the app cleans at inference and
training does not, the model is trained on a slightly different distribution
than it is asked to read. On the current 43 samples it makes no difference —
they were all written with pen lifts, so cleanup is a no-op on them — which is
why it has been left alone rather than changed blind. Revisit it if sloppy
samples get collected and fine-tuning becomes viable (it needs 150-300+ anyway).

**This is not the cursive fix.** It keeps ligatures intact and clears the
non-writing marks out of the way, which is a prerequisite for any recognizer —
a stroke model chokes on a drag exactly as an image model does. Reading joined,
sloppy handwriting well is still phase 3.

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
0. `python -m scripts.inspect_cleanup --dump-png out/` — confirm ink cleanup
   removes ~0% on the existing (pen-lifted) samples, then collect a few written
   without lifting the pen and re-run. See *Ink cleanup* above.
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
- Tests: `python -m pytest` (121, all passing)
