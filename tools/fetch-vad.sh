#!/usr/bin/env bash
# fetch-vad.sh — download Silero VAD v4 ONNX into models/vad/
# Re-runnable; downloads to a deterministic path.
set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/../models/vad" && pwd)"
DEST_FILE="$DEST_DIR/silero_vad_v4.onnx"
URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_FILE" ]]; then
  echo "fetch-vad: $DEST_FILE already present, skipping"
  exit 0
fi

# Silero ships a torch.jit blob, not an ONNX file. The daemon links it
# directly via libtorch. If you want the ONNX export, run:
#   python3 -c "import torch, onnx; m=torch.jit.load('$DEST_FILE'); torch.onnx.export(m, torch.randn(1,512), '$DEST_FILE.onnx', opset_version=17)"
# then point VAD_MODEL_PATH at the .onnx file.
echo "fetch-vad: downloading $URL -> $DEST_FILE"
curl -fL "$URL" -o "$DEST_FILE"
echo "fetch-vad: done ($(du -h "$DEST_FILE" | cut -f1))"
