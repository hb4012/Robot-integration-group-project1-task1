#!/bin/bash

set -Eeuo pipefail
export PATH="/home/nvidia/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$PROJECT_DIR/ros2_ws"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/best.pt}"
CAMERA_SOURCE="${1:-0}"
DISPLAY_OUTPUT="${DISPLAY_OUTPUT:-true}"
CONFIDENCE="${CONF:-0.35}"
IOU_THRESHOLD="${IOU:-0.45}"
IMAGE_SIZE="${IMGSZ:-640}"
ROS_SETUP=""

for candidate in /opt/ros/*/setup.bash; do
  if [[ -f "$candidate" ]]; then ROS_SETUP="$candidate"; break; fi
done
if [[ -z "$ROS_SETUP" ]]; then
  echo "错误：/opt/ros下没有找到ROS2。" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

if ! command -v ros2 >/dev/null 2>&1 || ! command -v colcon >/dev/null 2>&1; then
  echo "Error: ROS 2 or colcon is unavailable." >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: model file not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ "$DISPLAY_OUTPUT" != "true" && "$DISPLAY_OUTPUT" != "false" ]]; then
  echo "Error: DISPLAY_OUTPUT must be true or false." >&2
  exit 1
fi
if [[ "$DISPLAY_OUTPUT" == "true" ]]; then
  export DISPLAY="${DISPLAY:-:0}"
fi
if ! "$PYTHON_BIN" -c "import cv2, torch, ultralytics, rclpy" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN cannot import cv2, torch, ultralytics, or rclpy." >&2
  exit 1
fi

cd "$WORKSPACE" || exit 1
PYTHONNOUSERSITE=1 colcon build --symlink-install --packages-select desktop_object_detector || exit 1
set +u
# shellcheck disable=SC1091
source "$WORKSPACE/install/setup.bash"
set -u

REQUESTED_DEVICE="${DEVICE:-auto}"
DEVICE="$REQUESTED_DEVICE"
if [[ "$REQUESTED_DEVICE" == "auto" ]]; then
  DEVICE="cpu"
  if "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    DEVICE="0"
  fi
fi

RUN_DIR="$PROJECT_DIR/results/ros2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
LOGGER_PID=""
CLEANED_UP=0

cleanup() {
  if (( CLEANED_UP )); then
    return
  fi
  CLEANED_UP=1
  if [[ -n "$LOGGER_PID" ]] && kill -0 "$LOGGER_PID" >/dev/null 2>&1; then
    kill -INT "$LOGGER_PID" >/dev/null 2>&1 || true
    for _ in {1..30}; do
      if ! kill -0 "$LOGGER_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$LOGGER_PID" >/dev/null 2>&1; then
      kill -TERM "$LOGGER_PID" >/dev/null 2>&1 || true
    fi
    wait "$LOGGER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ros2 run desktop_object_detector result_logger --ros-args \
  -p "output_path:=$RUN_DIR/detections.jsonl" &
LOGGER_PID=$!
sleep 0.5
if ! kill -0 "$LOGGER_PID" >/dev/null 2>&1; then
  echo "Error: the ROS 2 result logger failed to start." >&2
  wait "$LOGGER_PID" || true
  exit 1
fi

echo "ROS 2 detection started"
echo "  Model:       $MODEL_PATH"
echo "  Source:      $CAMERA_SOURCE"
echo "  Device:      $DEVICE"
echo "  Detections:  /desktop_detector/detections"
echo "  FPS:         /desktop_detector/fps"
echo "  Result path: $RUN_DIR"
if [[ "$DISPLAY_OUTPUT" == "true" ]]; then
  echo "Press q in the detector window to stop."
else
  echo "Headless mode is enabled; press Ctrl+C to stop."
fi

set +e
ros2 run desktop_object_detector detector_node --ros-args \
  -p "model_path:=$MODEL_PATH" \
  -p "source:='$CAMERA_SOURCE'" \
  -p "device:='$DEVICE'" \
  -p "confidence:=$CONFIDENCE" \
  -p "iou:=$IOU_THRESHOLD" \
  -p "imgsz:=$IMAGE_SIZE" \
  -p "display:=$DISPLAY_OUTPUT"
STATUS=$?
set -e
exit "$STATUS"
