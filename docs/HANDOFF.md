# Handoff — Handwriting → Text

Touchscreen handwriting recognition for a Raspberry Pi 5. Write on the screen
with a finger or stylus, get editable text. Fully offline.

**Repo:** github.com/evanmettlen-glitch/Handwriting-App (public)
**Hardware:** Raspberry Pi 5 "HomeHubSSD", HDMI display + USB touch panel
**State as of 2026-09-03:** working end to end. **Latency is solved**
(`--model-dir microsoft/trocr-small-handwritten`: 4.21s → 0.62s/line, for ~14%
relatively worse CER). **Accuracy is now the open problem** — 0.425 CER and
13/37 exact on natural-language samples: single words are perfect, multi-word
lines frequently are not. Three real bugs found by testing on the Pi this
session and fixed — ink cleanup destroying cursive letters, `--quantize`
crashing on ARM, and a benchmark that measured accuracy on an unrepresentative
4-sample slice. See *Ink cleanup* and *Latency* before touching either area.
27 commits, 132 tests passing (117 on the Pi itself; 15 need a display and are
skipped over SSH).

---

## Start here

```bash
ssh evmett@HomeHubSSD
cd ~/HandWritingApp
./run.sh --model-dir microsoft/trocr-small-handwritten   # 6.8x faster, ~14% worse CER
# add --fullscreen for kiosk mode
```

Startup (model load + warm-up) measured **~6s** on the Pi for the default
`trocr-base` (0.8s for small), then ~4.2s/line at fp32 vs **0.62s** for small.
That speed comes at a real cost — natural-language CER 0.425 → 0.486 over the
full sample set — so pick deliberately; *Latency* below has both tables. The
status line at the bottom tells you exactly what is active:

```
Ready · backend: trocr-torch:trocr-small-handwritten
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

### Latency — solved 2026-09-03, was the open problem as of 2026-08-31

**Time to result was the complaint; it is now fixed.** The whole cost is TrOCR
on a CPU — render, cleanup, segmentation, and spell correction are all
sub-millisecond. Measure with `python -m scripts.bench_latency`, which reports
throttle state, a per-stage breakdown, and a speed-vs-CER sweep.

Note the *accuracy* claim in the 2026-08-31 version of this section ("accuracy
is now acceptable") did not survive being measured over the full sample set —
see the accuracy table below. Speed is solved; accuracy is not.

Shipped, no accuracy risk:

- **Startup warm-up.** The first inference used to cost ~30 s of lazy init on
  the user's first real line. `Recognizer.warmup()` now runs a throwaway pass
  during load, before the Recognize button enables.
- **A counting-up status line** (`Recognizing…  2.4s`) and elapsed time on the
  result, so a slow run reads as slow rather than hung.

**Measured on the Pi, 2026-09-03** — 8 samples, greedy decoding, temp 49°C,
throttle clean:

**Speed**, from the sweep (8-sample subset — see the CER caveat below):

| Config | Median | Verdict |
|---|---|---|
| base, 384px, fp32 | 4.21s | baseline |
| base, 224px, fp32 | 3.25s | rejected on accuracy |
| base, 384px, int8 | 3.84s | rejected — output is garbage, see below |
| small, 384px, fp32 | **0.62s** | **6.8x faster** |
| small, 224px, fp32 | 0.44s | rejected on accuracy |
| small, 384px, int8 | 0.81s | rejected — *slower* than fp32 |

**Accuracy**, from `eval_backend` over **all 43 samples** (37 natural-language,
6 rote) — this is the number to trust:

| Model | natural-language CER (n=37) | exact | rote CER (n=6) |
|---|---|---|---|
| trocr-base-handwritten | **0.425** | 13/37 (35%) | 0.808 |
| trocr-small-handwritten | **0.486** | 13/37 (35%) | 0.929 |

**The model swap is a real trade, not a free win.** 6.8x faster for ~14%
relatively worse CER, same count of exactly-correct lines. Defensible for a
notes pad; not defensible to describe as free.

> ⚠️ **A measurement trap that already produced one wrong conclusion — read
> this before trusting any `--limit`ed run.** This section originally claimed
> "CER 0.000 on both, zero accuracy cost". That came from
> `bench_latency --limit 8`, which sliced the *first* 8 samples. Enrollment
> order is graded — rote coverage, then single words, then multi-word lines —
> so those 8 were 4 rote (excluded from CER by design) plus `the`, `and`,
> `you`, `was`. The accuracy column was averaging **four one-word samples**
> while all 21 multi-word samples, the ones that actually fail, went
> unmeasured. `bench_latency` now uses `representative()` to spread picks
> across the set, prints how many samples the CER covers, and warns loudly
> below 10 scored samples. The lesson generalises: on this dataset, *any*
> accuracy claim from a small limit is close to meaningless, because the
> difficulty is sorted.

**`--image-size` costs real accuracy, measured, not guessed.** ~25% faster
matches the patch-count arithmetic (577→197 patches), but CER goes to 0.5-0.9 on
both models. Kept in the code — a future checkpoint fine-tuned at that
resolution might not pay this cost — but not part of the recommended setup.

**`--quantize` is a net loss, and it was also hard-broken until this session.**
Every attempt used to fail with `RuntimeError: unknown architecure` (torch's own
typo) — a crash, not an accuracy tradeoff. Root cause:
`torch.backends.quantized.engine` defaults to `"x86"` on every platform,
including aarch64, and that dispatch has no kernel for ARM. `qnnpack` is right
there in `torch.backends.quantized.supported_engines` and never gets picked.
Fixed in `resolve_quantized_engine()` (`trocr_torch_recognizer.py`) — switches
to `qnnpack` only when the engine is still that unconfigured default and the
machine is actually ARM, so an x86 box is untouched; verified going from a hard
crash to `quantized ok: True`.

With that fixed, quantizing turned out not to help at all. On `small` it is
**slower** than fp32 (0.66s → 0.81s measured directly, not the table's rounded
0.62/0.81) for no CER change — the qnnpack dynamic-quant overhead outweighs
anything it saves on a model this size. On `base` it runs without crashing but
the output is wrong in a way fp32 never is on the same inputs — four words that
read perfectly at fp32:

```
'the' -> '8th q'
'and' -> 'car us of'
'you' -> '1/ MO.S'
'was' -> 'wrote .'
```

A console warning during quantization (`qnnpack incorrectly ignores
reduce_range when it is set to true`) is a plausible contributing factor —
reduce_range trims the effective int8 range to avoid overflow on certain ops,
and qnnpack silently not honoring it could explain damage concentrated in the
attention-heavy base model — but this is an observation, not a diagnosis; it
was not chased further because the practical answer (don't use `--quantize`
here) didn't need it. **Don't use `--quantize` on this hardware.**

Old table, kept for what changed and why — everything in it except `--beams 1`
turned out to need correcting once actually measured:

| Flag | Guessed effect | What measuring found |
|---|---|---|
| `--model-dir microsoft/trocr-small-handwritten` | ~5x less compute, "real accuracy risk" | 6.8x for ~14% worse CER — the risk was real, just smaller than feared |
| `--quantize` | ~2x faster | crashed on ARM until fixed; once fixed, slower-or-garbage |
| `--image-size 224` | ~3x less encoder work, "needs measuring" | measured: not worth it, see above |
| `--beams 1` | ~4x less decode — **already the default** | confirmed; the checkpoints ship `num_beams=4` |

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

`on_partial` is passed to `recognize()` only when a caller asks for partials.
The app always asks — `app.py` passes `on_partial=self._on_partial` on every
run — so in practice every backend receives it; tesseract and the ONNX path
accept it and simply never call it, since both return a whole line at once.
The "only when asked" path matters for direct `pipeline.run(ink)` callers
(`eval_backend`, `calibrate`), not for the app. `--no-predict` drops the
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

**Verified on real handwriting, 2026-09-03 — and it found a real bug.** The
thresholds shipped tuned on synthetic ink alone (simple caret-shaped "letters"
that never bent the way a real one does), with a note here saying so. Run
against the 43 real enrollment samples on the Pi:

```bash
python -m scripts.inspect_cleanup --dump-png out/
```

`min_length=1.0` visibly destroyed letters in 2 of 43 samples: the flat top of
the cursive 'e' bowl in "the" and the 't' crossbar in "they" both got cut,
leaving mangled shapes (`out/0005-*.png`, `out/0026-*.png` from that run — worth
regenerating and eyeballing after any future threshold change). Both measured
**1.02-1.08x the writing height** — comfortably past what had been the safe
length, and well beyond where any synthetic test letter ever reached, because a
wide cursive letter's connecting strokes are exactly the "flat, straight,
one-way" shape the detector is looking for. `--sweep` confirmed the fix: 1.0 was
unsafe (worst case lost 15.8% of a sample for zero benefit — no word count
improved), 1.2 was the first value with zero loss across all 43, and 1.5-3.0
tied it exactly. Shipped default is now **1.5** — a margin above the first safe
point rather than the sweep's own longest-tied pick, which would have thrown
away sensitivity to genuine drags for no measured gain on this data.
`fast_min_length` moved by the same ratio (0.5→0.75) for the same reason, though
it never fired on any of the 43 real samples — none were written without
lifting the pen between words, so that half is still unexercised by real data.
There is now a regression test (`test_a_wide_cursive_connector_just_over_one_line_height_survives`
in `tests/test_cleanup.py`) reproducing this exact shape and ratio, so it cannot
silently come back.

**What is still open.** No real no-lift-between-words or genuine-drag sample
exists in this dataset — collect a few (`./run.sh --train --freeform`, write a
phrase without lifting the pen, or drag a finger across the pad on purpose) and
re-run the sweep. Two numbers matter: word counts (cutting a real slide should
make `segment_words` agree with the label more often) and ink removed on a
sample whose word count did *not* improve (that is damage, full stop — the
`--dump-png` renders are the way to catch it, not the summary numbers alone).

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
0. **Done, 2026-09-03.** `inspect_cleanup` against the 43 real samples found
   `min_length=1.0` cutting real letters — fixed to 1.5, see *Ink cleanup*
   above. Still open: collect a few samples written without lifting the pen and
   re-run `--sweep` to validate the other half (`fast_min_length`), which no
   real sample has exercised yet.
1. **Done, 2026-09-03.** `bench_latency` swept model/beams/quantize/image-size
   on the Pi. Answer: `--model-dir microsoft/trocr-small-handwritten`, nothing
   else — 6.8x faster for ~14% relatively worse CER (0.425 -> 0.486 over the
   full 43-sample set; the first pass claimed "zero cost" from an
   unrepresentative 8-sample slice, since fixed). `--quantize` and
   `--image-size` both measured net-negative; see *Latency* above. Also fixed
   in passing: `--quantize` was hard-crashing on ARM (wrong torch backend
   engine), separately from being not worth using once it ran.
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
- Tests: `python -m pytest` (132, all passing)
