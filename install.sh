#!/usr/bin/env bash
# Set up the app on Raspberry Pi OS (Bookworm) / Debian.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing system packages"
sudo apt update
sudo apt install -y python3-tk python3-venv python3-pip tesseract-ocr libopenjp2-7

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

cat <<'EOF'

Done.

  Run it:            ./run.sh
  Fullscreen kiosk:  ./run.sh --fullscreen

Optional higher-accuracy neural backend:
  ./.venv/bin/pip install -r requirements-trocr.txt
  ./.venv/bin/python -m scripts.export_trocr_onnx
  ./run.sh --backend trocr
EOF
