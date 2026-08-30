# Handwriting → Text

A kiosk-style app for the Raspberry Pi 5: write on a touchscreen with a finger or
stylus and it converts your handwriting to editable text. Runs fully offline.

Works with **any HDMI display + USB touch panel** that the Pi already sees as a
pointer device — which is almost all of them, with no extra drivers.

## How it works

```
finger / stylus ─▶ USB touch panel (evdev → pointer events)
                        │
                 InkCanvas (Tkinter)     captures strokes as (x, y) polylines
                        │
                 Ink.render()            rasterizes strokes → clean B/W image
                        │
                 Recognizer backend      image → text   (on a worker thread)
                        │
                 Text box                append · edit · copy to clipboard
```

Two interchangeable recognition backends:

| Backend                | Weight              | Accuracy                                   | Setup                      |
|------------------------|---------------------|--------------------------------------------|----------------------------|
| `tesseract` (default)  | tiny (apt package)  | good for neat block printing; weak cursive | none                       |
| `trocr`                | large (~1 GB)       | strong on messy print & some cursive       | one-time model export      |

`trocr` uses Microsoft's TrOCR handwritten model on ONNX Runtime; expect
~1–3 s per line on the Pi 5 CPU.

## Install

```bash
git clone <this-repo> ~/HandWritingApp
cd ~/HandWritingApp
./install.sh
./run.sh
```

`install.sh` installs `python3-tk` and `tesseract-ocr`, creates a `.venv`, and
installs Pillow.

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

### Flags

```
--backend {tesseract,trocr}
--fullscreen                start in kiosk mode
--no-auto                   manual recognition only
--auto-delay MS             pause before auto-recognize (default 1200)
--stroke-width PX            pen thickness (default 8)
--font-scale N              enlarge all UI text (e.g. 1.4 on small hi-dpi panels)
--lang eng+deu              Tesseract languages (needs tesseract-ocr-deu, etc.)
--psm N                     Tesseract segmentation (7 = one line, 6 = block)
--whitelist 0123456789      restrict recognized characters
--keep-ink                  don't clear the pad after each recognition
```

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

- **Use `--backend trocr`.** `tesseract` cannot read normal handwriting well and
  no amount of tuning changes that — it was built for scanned print.
- Write large, upright, well-spaced characters near the baseline guide.
- With `tesseract`: one word at a time, and try `--psm 8` (single word) or
  `--psm 13` (raw line) if `--psm 7` misreads.
- Use `--whitelist` when you only need digits or A–Z.

## Tests

```bash
./.venv/bin/pip install pytest
./.venv/bin/python -m pytest
```

## Limits

- Not true online recognition — strokes are rasterized then image-OCR'd; stroke
  order, timing and pressure aren't used.
- `tesseract` is an OCR engine, not a handwriting model — expect errors on
  anything that isn't tidy printing.
- TrOCR-small is line-level; split very long lines.

## Layout

```
handwriting_app/
  app.py                     Tkinter UI + threading
  canvas_widget.py           stroke capture
  ink.py                     stroke model + rasterization
  config.py                  CLI args
  recognizer/
    base.py                  Recognizer interface + RecognitionError
    tesseract_recognizer.py  default backend (subprocess)
    trocr_onnx_recognizer.py optional neural backend
scripts/export_trocr_onnx.py one-time ONNX export
systemd/handwriting-app.service
tests/
```
