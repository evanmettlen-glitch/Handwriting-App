# Handwriting → Text

A kiosk-style app for the Raspberry Pi 5: write on a touchscreen with a finger or
stylus and it converts your handwriting to editable text. Runs fully offline.

Works with **any HDMI display + USB touch panel** that the Pi already sees as a
pointer device — which is almost all of them, with no extra drivers.

## How it works

```
finger / stylus ─▶ USB touch panel (evdev → pointer events)
                        │
                 InkCanvas (Tkinter)     captures each pen-down..up as a stroke
                        │
                 clean_ink()             cut accidental drags and the slide
                        │                between words when the pen never lifts
                 segment_words()         group strokes into words by pen-lift gaps
                        │
                 Ink.render(deslant)     shear out slant, rasterize each word
                        │
                 Recognizer backend      image → text   (on a worker thread)
                        │
                 SpellCorrector          snap tokens to real English words
                        │
                 Text box                append · edit · copy to clipboard
```

Two interchangeable recognition backends:

| Backend      | Weight             | Accuracy                                     | Setup                    |
|--------------|--------------------|----------------------------------------------|--------------------------|
| `tesseract`  | tiny (apt package) | OK for block printing only; poor on cursive  | none                     |
| `trocr`      | ~1 GB              | strong on messy print and cursive            | `pip install` only       |

`--backend auto` (the default) uses `trocr` when torch + transformers are
installed, otherwise `tesseract`. See [docs/RECOGNITION.md](docs/RECOGNITION.md)
for the research notes and the roadmap.

## Install

```bash
git clone <this-repo> ~/HandWritingApp
cd ~/HandWritingApp
./install.sh
./run.sh
```

`install.sh` installs `python3-tk` and `tesseract-ocr`, creates a `.venv`, and
installs Pillow + symspellpy (the English dictionary for output correction).

### Neural backend (needed for real handwriting)

`tesseract` is an OCR engine for printed text — it misreads most handwriting.
For anything other than careful block capitals, install TrOCR:

```bash
./.venv/bin/pip install -r requirements-trocr.txt
./run.sh
```

That's it — no export step. The model (`microsoft/trocr-base-handwritten`)
downloads and caches on first run, and `--backend auto` picks it up
automatically. Expect ~3–8 s/line on a Pi 5 CPU — or ~0.6s with
`--model-dir microsoft/trocr-small-handwritten`, which is measurably faster for
a measurably real accuracy cost (CER 0.425 → 0.486). See *Making it faster*
below for both numbers before picking one.

**Optional speed-up.** Exporting to ONNX roughly halves latency, and the app
prefers an exported model automatically when it finds one:

```bash
./.venv/bin/pip install "optimum[onnxruntime]"
./.venv/bin/python -m scripts.export_trocr_onnx \
    --model microsoft/trocr-base-handwritten \
    --out models/trocr-base-handwritten-onnx --quantize
```

**Currently unusable on this Pi:** `optimum` is incompatible with torch 2.13
(`_attention_scale` was removed from `torch.onnx.symbolic_opset14`), so the
export above fails. The torch path needs no export and is what everything here
is measured on — treat ONNX as a lead to revisit, not a step to follow.

## Using it

- Write a word or short phrase in the pad.
- Pause — it auto-recognizes and appends to the text box (toggle **Auto**, or `--no-auto`).
- Write sloppily if you like: a finger dragged across the pad, a knuckle tap off
  to the side, or a whole phrase written without lifting the pen are all removed
  before recognition, and the status line says what went (`· 1 drag cut`). See
  *Sloppy input* below, or `--no-cleanup` to keep every mark.
- Or tap **Recognize**.
- **↶ Undo** drops the last stroke (`Ctrl+Z`) — one stray mark shouldn't cost
  you the whole pad. **Clear pad** wipes it (`Ctrl+L`).
- **Space / ⌫ / ↵** edit the output; the box is also directly editable with a keyboard.
- **Copy all** puts the text on the clipboard.
- **Exit** quits the app (or `Ctrl+Q`). `F11` toggles fullscreen, `Esc` leaves fullscreen.
  `Ctrl+Return` recognizes now without waiting for the pause.
- Teach it your own handwriting with **`./run.sh --train`** — see *Training mode* below.

### Flags

```
--backend {auto,tesseract,trocr}
--fullscreen                start in kiosk mode
--no-auto                   manual recognition only
--auto-delay MS             pause before auto-recognize (default 1800)
--stroke-width PX            pen thickness (default 8)
--font-scale N              enlarge all UI text (e.g. 1.4 on small hi-dpi panels)

personalization:
--user NAME                  load models/NAME-onnx + NAME's learned word list
--model-dir DIR             force a specific model directory
--no-personal-lexicon       ignore words learned from collected samples
--no-calibration            ignore data/samples/calibration.json

recognition pipeline:
--segment / --no-segment    word-by-word vs whole-line (default: whole line for
                            TrOCR, word-by-word for tesseract)
--word-gap-ratio R          word-break gap ÷ writing height (default 0.4)
--no-cleanup                keep every mark, including drags and stray taps
--no-predict                don't guess the rest of the word in the live preview
--no-deslant                keep slanted writing as-is
--no-smooth                 render strokes as straight lines, not splines
--no-spellcheck             don't correct output against the English dictionary
--no-join-letters           don't glue "a n d" back into "and"
--spell-compound            aggressive dictionary pass; also fixes bad spacing

recognition speed (TrOCR — see "Making it faster"):
--beams N                   beam width (default 1 = greedy; the checkpoints
                            ship 4, which costs ~4x the decode time)
--quantize                  load the model as dynamic int8 (measured slower
                            on this Pi — see "Making it faster")
--image-size PX             run the vision encoder at PX by PX instead of 384
                            (224 is ~3x less encoder work; measure the accuracy)
--max-tokens N              cap on characters generated per line (default 48)

tesseract backend:
--lang eng+deu              languages (needs tesseract-ocr-deu, etc.)
--psm N                     page segmentation mode, applied to every image
                            (default: 8 for a word, 7 for a line)
--whitelist 0123456789      restrict recognized characters

--keep-ink                  don't clear the pad after each recognition
```

## Training mode — teach it your handwriting

A personal model is the biggest accuracy win.

```bash
./run.sh --train
```

This runs a **guided enrollment**: ~40 short prompts (rote alphabet, a pangram,
digits, punctuation, a few sentences) ordered for full letter/digit coverage in
**under 5 minutes**. The header shows a progress bar to 100%, elapsed time, an
estimate of time left, and live `a-z / A-Z / 0-9` coverage; it marks you
**enrolled ✓** once there's enough. `Return` or `Space` = Save & next.
Samples land in `data/samples/` and resume automatically.

`--freeform` (or `--prompts-file FILE`) switches to an open-ended word list for
collecting more; `--enroll-target N` changes what counts as 100%.
`--user NAME` keeps each person's samples in their own `data/samples/NAME/`
folder (re-running resumes where that user left off).

Then, on a machine with a GPU (the Pi only does inference):

```bash
pip install -r requirements-train.txt
python -m scripts.finetune_trocr --dry-run                 # sanity-check the data
python -m scripts.finetune_trocr --out models/trocr-personal
python -m scripts.export_trocr_onnx --model models/trocr-personal \
    --out models/trocr-personal-onnx --quantize
```

Or do both steps at once (runs on the Pi too, ~20–40 min on CPU):

```bash
./scripts/train_personal.sh            # data/samples -> models/trocr-personal-onnx
./scripts/train_personal.sh evan       # data/samples/evan -> models/evan-onnx
```

> **Fine-tuning needs ~150–300+ samples.** On 40 it overfits and gets *worse*
> than the stock model (measured: val CER 0.52 → 0.80). With a 5-minute
> enrollment, use calibration below instead — it's the same idea Apple uses:
> keep one strong general model and adapt around it rather than retraining it.

### Sloppy input — drags, no pen lift, stray marks

Not every mark on the pad is writing. A finger brushing the glass leaves a line
through the middle of a word; a knuckle drops a speck off to the side; writing a
phrase without lifting the pen records the slide from one word to the next as
ink. All three make the recognizer's job harder, and none of them are its fault.

`handwriting_app/cleanup.py` removes them before recognition. All three are the
same shape: a **traverse**, a long horizontal move that carries no vertical
information. Letters do not do that — even a cursive ligature is short and rises
and falls, while a traverse runs a line height or more and stays flat. Cutting a
traverse out also puts the missing pen lift back, so `segment_words()` can see
word boundaries again in writing that was never lifted.

The thresholds are fractions of the writing height, so they hold for large and
small hands alike, and they are set conservatively — deleting your writing is
much worse than leaving a drag in for the model to cope with. A shorter traverse
is only cut when it was also *fast* (the pen jumped several times further
between samples than it does while writing) *and* has real writing on both
sides, which together mean a pen that never lifted rather than a deliberate
dash. If cleanup would leave the pad empty it is abandoned instead.

Whatever it removes is named in the status line (`Added in 3.2s: 'hello world' ·
1 drag cut`), and `--no-cleanup` turns the whole thing off. Collected samples
always store the raw strokes — cleanup is a recognition-time step, so a change
to it applies retroactively to everything you have already recorded.

Check it against your own handwriting before trusting it:

```bash
python -m scripts.inspect_cleanup                  # what it removes, and whether it helped
python -m scripts.inspect_cleanup --dump-png out/  # before/after renders — look at them
python -m scripts.inspect_cleanup --sweep          # tune the length threshold
```

The number to watch is word counts: cutting the slide between two words should
make `segment_words` agree with the label more often. The number to *worry*
about is ink removed on a sample whose word count did not improve — that is a
letter going missing.

> **Cursive itself is a different problem.** TrOCR was trained on IAM, which is
> largely cursive, so joined writing is what it likes; the cleanup above keeps
> ligatures intact and only cuts genuine traverses. But properly robust cursive
> and sloppy writing needs a model that reads the pen *trajectory* rather than a
> bitmap — that is [docs/PHASE3_SCOPE.md](docs/PHASE3_SCOPE.md), not a threshold.

### Diagnosing bad accuracy

Before blaming the model, check that the touchscreen is giving you enough
points. Tk coalesces motion events, so a quick stroke can be captured as a
handful of widely-spaced samples:

```bash
python -m scripts.inspect_ink                 # sampling density per sample
python -m scripts.inspect_ink --dump-png out/ # then actually look at out/*.png
```

It reports two things and names the likely culprit:

- **sampling density** — if the median gap is more than a few pixels, strokes
  were rendering as straight-line polygons. `Ink.render(smooth=True)` (the
  default) interpolates a Catmull-Rom spline through the captured points to
  restore the curves; `--no-smooth` turns it off for comparison.
- **word segmentation** — it compares the number of segments found against the
  number of words in each label, and reports over-splitting separately from
  merging. **Over-splitting is the harmful one**: the model gets letter
  fragments instead of words. Merging into whole lines is fine for TrOCR — it
  was trained on IAM text lines and uses cross-word context, so a whole line is
  what it wants.

```bash
python -m scripts.inspect_ink --sweep     # then: ./run.sh --word-gap-ratio <best>
```

Segmentation defaults to **off for TrOCR** and **on for tesseract**; `--segment`
and `--no-segment` override. `scripts/calibrate.py` tunes the threshold
automatically when it applies, and it costs nothing — word counts need no
model inference.

### Measuring accuracy

```bash
python -m scripts.eval_backend            # one pass, prints per-sample results
python -m scripts.eval_backend --quiet    # summary only
```

Reports CER and exact-match accuracy, **split into natural-language prompts and
rote coverage prompts** (alphabet runs, digit strings). That split matters:
TrOCR's decoder is a language model, so it mangles `abcdefghijklm` however
neatly you wrote it. A single rote sample can drag an aggregate score from 0.05
to 0.75 — judge the model on the natural-language number.

Use this to check whether a change actually helped before keeping it.

### Making it faster

TrOCR-base on a Pi 5 CPU is the whole cost — everything else in the pipeline is
sub-millisecond. Measure before turning knobs:

```bash
python -m scripts.bench_latency
```

It prints the throttle/temperature state first (a throttled Pi looks exactly
like slow code, and no code change fixes that), then a per-stage breakdown —
render, preprocess, vision encoder, decode, postprocess — then a sweep of
median seconds *and* CER for each speed setting. Speed is only worth having if
the accuracy column holds, so the two are always reported together.

The levers, biggest first — **measured on the Pi, 2026-09-03**, 8 samples,
greedy decoding, temp 49°C with a clean throttle state so the numbers are real:

| Lever | Speed | Accuracy (CER over **all 37** natural-language samples) |
|---|---|---|
| `--model-dir microsoft/trocr-small-handwritten` | **4.21s → 0.62s** (6.8x) | 0.425 → **0.486** — a real cost, ~14% relative. Exact matches unchanged at 13/37 |
| `--beams 1` (**already the default**) | ~4x less decode vs. the checkpoints' default of 4 | negligible |
| `--image-size 224` | 4.21s → 3.25s on base | **don't** — badly worse |
| `--quantize` | 0.62s → 0.81s on small; 4.21s → 3.84s on base | **don't** — slower on small, and destroys base's output entirely |

**The model swap is a real trade, not a free win.** 6.8x faster for ~14%
relatively worse CER (0.425 → 0.486), with the same number of exactly-correct
lines (13/37). Whether that is worth it is a judgement call about how the pad is
used: for short notes you re-read anyway, 0.6s beats 4.2s comfortably; if you
need the best reading the model can give, stay on base.

> **How this number got corrected, because it matters.** The first version of
> this table claimed "CER 0.000 on both, no accuracy cost". That came from
> `bench_latency --limit 8`, which used to slice the *first* 8 samples — and
> enrollment order is graded, so those were 4 rote prompts (excluded from CER)
> plus `the`, `and`, `you`, `was`. The accuracy column was averaging four
> one-word samples while all 21 multi-word samples went unmeasured. Re-running
> over the full set gave the numbers above. `bench_latency` now spreads its
> picks across the set and prints how many samples the CER actually covers,
> with a loud warning under 10 — the tool made a wrong conclusion easy, so the
> tool was fixed too, not just the number.

**Accuracy is mediocre either way, and that is the real open problem.** 0.425
CER with 13/37 exact is not a good reading of this handwriting — single words
come out perfect, multi-word lines often do not (`'I will call you later'` →
`'You Need .'`). Speed is now fine; accuracy on multi-word lines is what is
left. See [docs/PHASE3_SCOPE.md](docs/PHASE3_SCOPE.md).

**`--image-size` is not worth it, measured.** The vision encoder's cost is
fixed per image and set by patch count — 384x384 is 577 patches, 224x224 is
197 — so the ~25% wall-clock win is real. But on the 8-sample subset it was
measured on, CER went from 0.000 to 0.917 on base and 0.000 to 0.500 on small.
That subset is the unrepresentative one described above, so treat the exact
figures as indicative — the *direction* is not in doubt (it made every sample
it touched worse, on both models), which is enough to rule it out. The
position-embedding interpolation it needs is not free on real handwriting.
Left in the code (`--image-size PX`, off by default) in case a checkpoint
fine-tuned at that resolution ever makes it viable.

**`--quantize` is a net loss on this hardware, and it was also outright broken
until this session.** Every attempt used to fail with `RuntimeError: unknown
architecure` (torch's own typo) — a crash, not an accuracy tradeoff. Root cause:
`torch.backends.quantized.engine` defaults to `"x86"` regardless of host
architecture, and that has no kernel for the Pi's aarch64.
`handwriting_app/recognizer/trocr_torch_recognizer.py` now switches to
`qnnpack` (the ARM engine, already listed in torch's own `supported_engines`,
just never selected) — but only when the current engine is still that broken
default, so an x86 dev machine is untouched. With that fixed, quantizing turned
out not to help at all: **slower** on the small model (0.66s → 0.81s, no CER
change) and, on base, it runs but the output is garbage —
`'the' → '8th q'`, `'and' → 'car us of'`, `'you' → '1/ MO.S'`, four words fp32
reads perfectly. Not a subtle tradeoff. Don't use `--quantize` on this hardware.

Sweep any of this yourself — the flags exist and are measured honestly, even
where the answer turned out to be "don't":

```bash
python -m scripts.bench_latency --models microsoft/trocr-base-handwritten microsoft/trocr-small-handwritten --quantize off on
```

The other 1.8 s is not the model at all: `--auto-delay` is how long the app
waits for you to stop writing before it starts. Lower it (`--auto-delay 800`)
if you write in short bursts.

The **first** recognition used to take ~30 s because the model initializes
lazily. That cost is now paid at startup — the status line says
`Warming up the model` and the **Recognize** button stays disabled until it's
done, so every recognition the user actually makes runs at steady-state speed.
While one is running the status line counts up (`Recognizing…  2.4s`) and the
result reports what it took.

### Watching it decode

The decoder produces one token at a time, so there is no reason to stare at a
spinner until the whole line lands. The status line streams the text as it
arrives, and guesses where the word being decoded is heading:

```
Recognizing…  2.4s   ▸ the quick brow(n)
```

The guess in parentheses is completed from the English dictionary and your
personal word list, preferring your own vocabulary. It is **display only** —
what gets committed to the text box is always what the model actually produced,
never the guess. `--no-predict` drops the parenthesised half; the streaming
itself has no cost, because the tokens were being generated anyway.

To be clear about what this does and does not do: streaming does not make
recognition faster, it makes the wait legible, and lets you see a wrong reading
early enough to stop waiting for it. The seconds themselves come off with the
levers above.

### Calibration — personalization in minutes, no training

```bash
python -m scripts.calibrate
```

One forward pass over your samples. It grid-searches the render settings that
read *your* hand best, mines words the recognizer reliably misreads for you, and
writes `data/samples/calibration.json` — which the app loads automatically. It
tunes against the *cleaned* ink, the same thing the app feeds the model
(`--no-cleanup` to tune against the raw strokes instead). Works
from ~20 samples. Prints baseline vs tuned CER so you can see the gain.

**Once `calibration.json` exists, it overrides `--stroke-width`,
`--word-gap-ratio`, `--no-deslant` and `--no-smooth` unconditionally** — even
when you pass one of those explicitly on the command line, silently, with
nothing in the status line to say so. `--no-calibration` is the only way to
make those flags take effect again; it is not just an opt-out, it is
occasionally a required one.

### Using the learned data

The app picks it up on its own — no flags:

- **Model** — `./run.sh` auto-loads `models/trocr-personal-onnx` if it exists
  (or `models/<user>-onnx` with `--user <name>`), else a generic model, else
  tesseract. The status line shows which (`trocr:trocr-personal-onnx`).
- **Personal word list** — every word you wrote during enrollment is fed to the
  spell corrector as a known term, so it stops "correcting" your names and
  jargon into dictionary words while still fixing real typos. The status line
  notes `personal lexicon: N words`. Works with no model training at all.
  Disable with `--no-personal-lexicon`.
- **Calibration** — `calibration.json` overrides the render settings and applies
  your word fixes. Status notes `calibrated on N samples`.

```bash
./run.sh                 # single user
./run.sh --user evan     # evan's model + evan's word list
```

Full rationale and the roadmap beyond this: [docs/RECOGNITION.md](docs/RECOGNITION.md).

## Run at boot (kiosk)

**Wayland — labwc** (newer Pi OS): add to `~/.config/labwc/autostart`
```
~/HandWritingApp/run.sh --fullscreen &
```

**Wayland — wayfire** (Bookworm): in `~/.config/wayfire.ini`
```
[autostart]
handwriting = ~/HandWritingApp/run.sh --fullscreen
```

**X11 / systemd**: edit paths in `systemd/handwriting-app.service`, then
```bash
sudo cp systemd/handwriting-app.service /etc/systemd/system/
sudo systemctl enable --now handwriting-app.service
```

## Touch not working?

Most USB panels need no setup. If touches don't register or land in the wrong spot:

- `libinput list-devices` (or `xinput list`) — confirm the panel is detected.
- **Wrong monitor**: map touch to the display — `xinput map-to-output <id> HDMI-1`
  on X11, or set the touch device's `output` in the compositor config on Wayland.
- **Offset / inverted axes**: apply a calibration matrix via `xinput set-prop`
  (X11) or `libinput` calibration in the compositor config (Wayland).

## Accuracy tips

- **Export and use `--backend trocr`.** `tesseract` cannot read normal
  handwriting well — it was built for scanned print. Everything below matters
  much less than this.
- **Keep letters within a word close together.** This matters more than
  neatness. TrOCR is trained on connected handwriting, so widely-spaced
  printing reads back as separate one-letter words (`and` → `a n d`). The app
  glues those back when they form a real word, but tighter letters avoid the
  problem outright — and make word gaps unambiguous.
- Write near the baseline guide; size and slant are handled automatically.
- Best results come from a personal fine-tune — see
  [docs/RECOGNITION.md](docs/RECOGNITION.md), phase 2.
- `tesseract` only: try `--psm 13`, and `--whitelist` if you need just digits/A–Z.

## Tests

```bash
./.venv/bin/pip install pytest
./.venv/bin/python -m pytest
```

## Limits

- Not true online recognition yet — strokes are rasterized then image-OCR'd, so
  stroke order and timing aren't used by the model (only for word segmentation).
  The online stroke model is phase 3 in the roadmap.
- `tesseract` is an OCR engine, not a handwriting model.
- TrOCR is line/word-level; the pipeline handles one line at a time.

## Layout

```
handwriting_app/
  app.py                     Tkinter UI + threading
  canvas_widget.py           stroke capture
  ink.py                     stroke model, spline smoothing, deslant, raster
  segmentation.py            group strokes into words by pen-lift gaps
  pipeline.py                segment → recognize → dictionary-correct
  postprocess.py             SymSpell English-word correction
  config.py                  CLI args
  training.py                --train UI: guided enrollment, progress bar, timer
  enrollment.py              curated <5 min prompt set + coverage tracking
  widgets.py                 ProgressBar
  dataset.py                 read/write samples under data/samples/
  prompts.py + data/prompts.txt   freeform word list
  models.py                  auto-discover the model dir (personal > generic)
  lexicon.py                 build a personal word list from sample labels
  calibration.py             calibration.json: render settings + word fixes
  textalign.py               CER + character confusion alignment
  naming.py                  user-name -> safe path slug
  recognizer/
    base.py                  Recognizer interface + RecognitionError
    tesseract_recognizer.py  fallback backend (subprocess)
    trocr_torch_recognizer.py neural backend, plain torch (no export needed)
    trocr_onnx_recognizer.py  same model via ONNX Runtime (faster, optional)
scripts/export_trocr_onnx.py one-time ONNX export (+ --quantize)
scripts/finetune_trocr.py    fine-tune on your samples (dev machine / GPU)
scripts/train_personal.sh    fine-tune + export in one command
scripts/calibrate.py         no-training personalization -> calibration.json
scripts/eval_backend.py      measure CER / accuracy on labelled samples
scripts/inspect_ink.py       stroke-capture density diagnostics
docs/RECOGNITION.md          research notes and roadmap
systemd/handwriting-app.service
tests/
```
