#!/usr/bin/env bash
# Fine-tune + export a personal model from collected samples, in one step.
#
#   ./scripts/train_personal.sh [name] [samples-dir]
#
# Examples:
#   ./scripts/train_personal.sh                 # data/samples -> models/personal
#   ./scripts/train_personal.sh evan            # data/samples/evan -> models/evan
#
# Runs anywhere Python + requirements-train.txt are installed. A CUDA GPU is
# strongly preferred; on a Pi 5 CPU expect 20-40 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-trocr-personal}"
SAMPLES="${2:-data/samples/$NAME}"
[ -d "$SAMPLES" ] || SAMPLES="data/samples"

PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "==> Samples:  $SAMPLES"
echo "==> Output:   models/$NAME  ->  models/$NAME-onnx"
"$PY" -m scripts.finetune_trocr --samples "$SAMPLES" --out "models/$NAME"
"$PY" -m scripts.export_trocr_onnx \
    --model "models/$NAME" --out "models/$NAME-onnx" --quantize

echo
if [ "$NAME" = "trocr-personal" ]; then
    echo "Done. The app now picks this up automatically:  ./run.sh"
else
    echo "Done. Use it:  ./run.sh --user $NAME"
fi
