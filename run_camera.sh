#!/bin/bash

set -Eeuo pipefail
export PATH="/home/nvidia/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export DISPLAY="${DISPLAY:-:0}"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CAMERA_SOURCE="${1:-0}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/best.pt}"
RUN_DIR="$PROJECT_DIR/results/live_$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "错误：找不到模型：$MODEL_PATH" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import cv2, torch, ultralytics" >/dev/null 2>&1; then
  echo "错误：Python无法导入cv2、torch或ultralytics。" >&2
  echo "请先运行：/bin/bash $PROJECT_DIR/check_jetson.sh" >&2
  exit 1
fi

echo "模型：$MODEL_PATH"
echo "摄像头：$CAMERA_SOURCE"
echo "保存目录：$RUN_DIR"
echo "窗口中按q结束。"

exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/detect.py" \
  --model "$MODEL_PATH" \
  --source "$CAMERA_SOURCE" \
  --device auto \
  --conf "${CONF:-0.35}" \
  --imgsz "${IMGSZ:-640}" \
  --output "$RUN_DIR" \
  --save-video \
  --save-frames "${SAVE_FRAMES:-120}"
