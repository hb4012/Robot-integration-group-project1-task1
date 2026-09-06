#!/bin/bash

set -uo pipefail
export PATH="/home/nvidia/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export DISPLAY="${DISPLAY:-:0}"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$PROJECT_DIR/ros2_ws"
MODEL_PATH="$PROJECT_DIR/models/best.pt"
CAMERA_SOURCE="${1:-0}"
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
  echo "错误：ROS2或colcon不可用。" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "错误：找不到模型：$MODEL_PATH" >&2
  exit 1
fi
if ! python3 -c "import cv2, torch, ultralytics, rclpy" >/dev/null 2>&1; then
  echo "错误：ROS2使用的python3缺少cv2、torch、ultralytics或rclpy。" >&2
  exit 1
fi

cd "$WORKSPACE" || exit 1
PYTHONNOUSERSITE=1 colcon build --symlink-install --packages-select desktop_object_detector || exit 1
set +u
# shellcheck disable=SC1091
source "$WORKSPACE/install/setup.bash"
set -u

DEVICE="cpu"
if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  DEVICE="0"
fi

RUN_DIR="$PROJECT_DIR/results/ros2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
LOGGER_PID=""

cleanup() {
  if [[ -n "$LOGGER_PID" ]]; then
    kill "$LOGGER_PID" >/dev/null 2>&1 || true
    wait "$LOGGER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

ros2 run desktop_object_detector result_logger --ros-args \
  -p "output_path:=$RUN_DIR/detections.jsonl" &
LOGGER_PID=$!
sleep 1

echo "ROS2识别已启动："
echo "  检测话题 /desktop_detector/detections"
echo "  FPS话题  /desktop_detector/fps"
echo "  保存目录  $RUN_DIR"
echo "窗口中按q结束。"

ros2 run desktop_object_detector detector_node --ros-args \
  -p "model_path:=$MODEL_PATH" \
  -p "source:='$CAMERA_SOURCE'" \
  -p "device:='$DEVICE'" \
  -p "confidence:=${CONF:-0.35}" \
  -p "imgsz:=${IMGSZ:-640}" \
  -p "display:=true"
STATUS=$?
cleanup
LOGGER_PID=""
exit "$STATUS"
