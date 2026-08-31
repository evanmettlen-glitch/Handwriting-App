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
automatically. Expect ~3–8 s/line on a Pi 5 CPU.

**Optional speed-up.** Exporting to ONNX roughly halves latency, and the app
prefers an exported model automatically when it finds one:

```bash
./.venv/bin/pip install "optimum[onnxruntime]"
./.venv/bin/python -m scripts.export_trocr_onnx \
    --model microsoft/trocr-base-handwritten \
    --out models/trocr-base-handwritten-onnx --quantize
```

Skip this if `optimum` fights your torch version — it pins awkwardly against
newer releases, and the torch path works fine without it.

## Using it

- Write a word or short phrase in the pad.
- Pause — it auto-recognizes and appends to the text box (toggle **Auto**, or `--no-auto`).
- Or tap **Recognize**.
- **Space / ⌫ / ↵** edit the output; the box is also directly editable with a keyboard.
- **Copy all** puts the text on the clipboard.
- **Exit** quits the app (or `Ctrl+Q`). `F11` toggles fullscreen, `Esc` leaves fullscreen.
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
--no-segment                recognize the whole line at once, not word by word
--word-gap-ratio R          word-break gap ÷ writing height (default 0.4)
--no-deslant                keep slanted writing as-is
--no-smooth                 render strokes as straight lines, not splines
--no-spellcheck             don't correct output against the English dictionary
--spell-compound            aggressive dictionary pass; also fixes bad spacing

tesseract backend:
--lang eng+deu              languages (needs tesseract-ocr-deu, etc.)
--psm N                     line segmentation (7 = one line, 13 = raw line)
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
  number of words in each label. Printed handwriting has inter-letter gaps that
  can rival inter-word gaps, so words get chopped into fragments and the model
  is asked to read single letters as words. Find a better threshold with:

```bash
python -m scripts.inspect_ink --sweep     # then: ./run.sh --word-gap-ratio <best>
```

`scripts/calibrate.py` tunes this automatically, and it costs nothing — word
counts need no model inference.

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

### Calibration — personalization in minutes, no training

```bash
python -m scripts.calibrate
```

One forward pass over your samples. It grid-searches the render settings that
read *your* hand best, mines words the recognizer reliably misreads for you, and
writes `data/samples/calibration.json` — which the app loads automatically. Works
from ~20 samples. Prints baseline vs tuned CER so you can see the gain.
Opt out at runtime with `--no-calibration`.

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
- Leave a clear space between words so the pen-lift segmenter can split them
  (tune with `--word-gap-ratio`).
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
