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

| Backend      | Weight             | Accuracy                                     | Setup                 |
|--------------|--------------------|----------------------------------------------|-----------------------|
| `tesseract`  | tiny (apt package) | OK for block printing only; poor on cursive  | none                  |
| `trocr`      | large (~1 GB)      | strong on messy print and cursive            | one-time model export |

`--backend auto` (the default) uses `trocr` when its model has been exported,
otherwise `tesseract`. See [docs/RECOGNITION.md](docs/RECOGNITION.md) for the
research notes and the roadmap (personal fine-tuning, online stroke model).

## Install

```bash
git clone <this-repo> ~/HandWritingApp
cd ~/HandWritingApp
./install.sh
./run.sh
```

`install.sh` installs `python3-tk` and `tesseract-ocr`, creates a `.venv`, and
installs Pillow + symspellpy (the English dictionary for output correction).

### Optional: neural backend (recommended for real handwriting)

`tesseract` is an OCR engine for printed text — it misreads most handwriting.
For anything other than careful block capitals, use TrOCR:

```bash
./.venv/bin/pip install -r requirements-trocr.txt
./.venv/bin/python -m scripts.export_trocr_onnx      # small model, downloads + converts once
./run.sh --backend trocr
```

For higher accuracy (slower, ~3-5 s/line on a Pi 5) export the base model:

```bash
./.venv/bin/python -m scripts.export_trocr_onnx \
    --model microsoft/trocr-base-handwritten \
    --out models/trocr-base-handwritten-onnx
./run.sh --backend trocr --model-dir models/trocr-base-handwritten-onnx
```

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

recognition pipeline:
--no-segment                recognize the whole line at once, not word by word
--word-gap-ratio R          word-break gap ÷ writing height (default 0.4)
--no-deslant                keep slanted writing as-is
--no-spellcheck             don't correct output against the English dictionary
--spell-compound            aggressive dictionary pass; also fixes bad spacing

tesseract backend:
--lang eng+deu              languages (needs tesseract-ocr-deu, etc.)
--psm N                     line segmentation (7 = one line, 13 = raw line)
--whitelist 0123456789      restrict recognized characters

--keep-ink                  don't clear the pad after each recognition
```

## Training mode — teach it your handwriting

A personal model is the biggest accuracy win. Collect samples of your own hand,
then fine-tune TrOCR on them.

```bash
./run.sh --train                 # shows a prompt, you write it, Save & next
```

Aim for 150–300 samples (progress and resume are automatic; samples land in
`data/samples/`). Then, on a machine with a GPU (the Pi only does inference):

```bash
pip install -r requirements-train.txt
python -m scripts.finetune_trocr --dry-run                 # sanity-check the data
python -m scripts.finetune_trocr --out models/trocr-personal
python -m scripts.export_trocr_onnx --model models/trocr-personal \
    --out models/trocr-personal-onnx --quantize
```

Back on the Pi:

```bash
./run.sh --model-dir models/trocr-personal-onnx
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
  ink.py                     stroke model + rasterization + deslant
  segmentation.py            group strokes into words by pen-lift gaps
  pipeline.py                segment → recognize → dictionary-correct
  postprocess.py             SymSpell English-word correction
  config.py                  CLI args
  training.py                --train UI: prompt → write → save sample
  dataset.py                 read/write samples under data/samples/
  prompts.py + data/prompts.txt   words/phrases to elicit
  recognizer/
    base.py                  Recognizer interface + RecognitionError
    tesseract_recognizer.py  default backend (subprocess)
    trocr_onnx_recognizer.py neural backend (ONNX Runtime)
scripts/export_trocr_onnx.py one-time ONNX export (+ --quantize)
scripts/finetune_trocr.py    fine-tune on your samples (dev machine / GPU)
docs/RECOGNITION.md          research notes and roadmap
systemd/handwriting-app.service
tests/
```
