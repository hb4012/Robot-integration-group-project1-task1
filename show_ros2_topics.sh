#!/bin/bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP=""

for candidate in /opt/ros/*/setup.bash; do
  if [[ -f "$candidate" ]]; then
    ROS_SETUP="$candidate"
    break
  fi
done

if [[ -z "$ROS_SETUP" ]]; then
  echo "Error: no ROS 2 installation was found under /opt/ros." >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/ros2_ws/install/setup.bash" ]]; then
  echo "Error: the ROS 2 workspace has not been built. Run ./run_ros2.sh first." >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1091
source "$PROJECT_DIR/ros2_ws/install/setup.bash"
set -u

echo "Available ROS 2 topics:"
ros2 topic list -t
echo
echo "Detector topic details:"
ros2 topic info /desktop_detector/detections || true
ros2 topic info /desktop_detector/fps || true
echo
echo "Use one of these commands to inspect live messages:"
echo "  ros2 topic echo /desktop_detector/detections"
echo "  ros2 topic echo /desktop_detector/fps"
