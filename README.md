# Running Instructions

## 1. Project Overview

This project runs a custom YOLO11n object detector on NVIDIA Jetson and can publish the detection results through ROS 2 Humble. The runtime program reports the class name, confidence score, bounding box, and real-time FPS.

The submitted detector recognizes the following three target classes:

- `bottle`
- `mouse`
- `laptop`

The checkpoint also contains a `cup` label, but the current runtime scripts deliberately filter it out because it is not one of the three experiment targets.

## 2. Required Files

Run all commands from the project root. On the Jetson, the expected location is:

```text
/home/nvidia/bht
```

The following files are required for inference:

```text
bht/
├── models/best.pt
├── scripts/detect.py
├── scripts/runtime_utils.py
├── ros2_ws/src/desktop_object_detector/
├── run_camera.sh
├── run_ros2.sh
└── requirements.txt
```

## 3. Environment Check on Jetson

The tested environment uses NVIDIA Jetson Orin NX, Python 3.10, ROS 2 Humble, OpenCV, PyTorch with CUDA support, and Ultralytics.

Open a terminal on the Jetson and run:

```bash
cd /home/nvidia/bht

/usr/bin/python3 --version
/usr/bin/python3 -c "import cv2, torch, ultralytics; print('OpenCV:', cv2.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__)"

test -f models/best.pt && echo "Model found"
ls -l /dev/video* 2>/dev/null || true
```

If a Python package is missing, install the project dependencies with:

```bash
python3 -m pip install --user -r requirements.txt
```

On Jetson, keep the NVIDIA-provided CUDA-enabled PyTorch installation. Do not replace it with a generic CPU-only PyTorch package.

## 4. Run Camera Detection on Jetson

Grant execution permission once:

```bash
cd /home/nvidia/bht
/bin/chmod +x run_camera.sh run_ros2.sh
```

Run the default camera at index `0`:

```bash
cd /home/nvidia/bht
/bin/bash ./run_camera.sh 0
```

If the USB camera is assigned another index, replace `0` with `1` or `2`:

```bash
/bin/bash ./run_camera.sh 1
```

Press `q` in the detection window to stop the program normally. The script uses `models/best.pt`, a 640-pixel input size, a confidence threshold of 0.35, and CUDA automatically when it is available.

Runtime settings can be changed temporarily through environment variables:

```bash
CONF=0.40 IMGSZ=640 SAVE_FRAMES=60 /bin/bash ./run_camera.sh 0
```

## 5. Run Detection Directly

The Python entry point can also process a Jetson camera, a video file, or an image stored on the Jetson.

Camera:

```bash
cd /home/nvidia/bht
python3 scripts/detect.py \
  --model models/best.pt \
  --source 0 \
  --device auto \
  --conf 0.35 \
  --imgsz 640 \
  --output results/manual_camera \
  --save-video \
  --save-frames 120
```

Video file:

```bash
python3 scripts/detect.py \
  --model models/best.pt \
  --source /home/nvidia/bht/input.mp4 \
  --device auto \
  --conf 0.35 \
  --output results/video_test \
  --save-video
```

Headless execution without a display window:

```bash
python3 scripts/detect.py \
  --model models/best.pt \
  --source 0 \
  --device auto \
  --conf 0.35 \
  --output results/headless_test \
  --save-video \
  --headless \
  --max-frames 600
```

## 6. Run with ROS 2

ROS 2 Humble and `colcon` must already be installed on the Jetson. Check them first:

```bash
source /opt/ros/humble/setup.bash
ros2 --help >/dev/null && echo "ROS 2 found"
colcon --help >/dev/null && echo "colcon found"
```

Start the detector, result publisher, and result logger:

```bash
cd /home/nvidia/bht
/bin/bash ./run_ros2.sh 0
```

The script automatically performs these operations:

1. Sources the installed ROS 2 environment.
2. Builds the `desktop_object_detector` package with `colcon`.
3. Starts the JSONL result logger.
4. Starts the YOLO detector node.
5. Publishes detections and FPS continuously.

To inspect the topics, open a second Jetson terminal and run:

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/bht/ros2_ws/install/setup.bash

ros2 topic list
ros2 topic echo /desktop_detector/detections
```

To view only the FPS stream:

```bash
ros2 topic echo /desktop_detector/fps
```

The ROS 2 topics are:

| Topic | Message type | Content |
| --- | --- | --- |
| `/desktop_detector/detections` | `std_msgs/msg/String` | JSON containing frame information, classes, confidence scores, and box coordinates |
| `/desktop_detector/fps` | `std_msgs/msg/Float32` | Real-time processing speed |

Press `q` in the detector window or `Ctrl+C` in the terminal to stop the ROS 2 run.

## 7. Result Files

Each normal camera run creates a time-stamped directory such as:

```text
results/live_YYYYMMDD_HHMMSS/
```

It may contain:

| File | Description |
| --- | --- |
| `annotated.mp4` | Detection video with boxes, labels, confidence scores, and FPS |
| `detections.csv` | One row per detected object, including frame ID, class, confidence, coordinates, inference time, and FPS |
| `summary.json` | Total frames, duration, mean FPS, class counts, and the 5 FPS acceptance result |
| `frames/` | Periodically saved annotated frames |
| `simultaneous_two_classes.jpg` | First frame in which at least two target classes are detected together |

Each ROS 2 run creates:

```text
results/ros2_YYYYMMDD_HHMMSS/
├── detections.jsonl
└── summary.json
```

`detections.jsonl` stores one published JSON message per line. `summary.json` records the total message count, frames containing detections, total boxes, and per-class counts.

## 8. Optional Dataset Validation and Training on Jetson

Validate the YOLO image-label pairs:

```bash
python3 scripts/check_dataset.py --data data/person1.yaml
```

Start YOLO11n transfer learning with the Jetson CUDA GPU:

```bash
python3 scripts/train.py \
  --model yolo11n.pt \
  --data data/person1.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0
```

Use `--device cpu` only when CUDA is unavailable. Training on the CPU will be considerably slower.

## 9. Troubleshooting

### `bash: command not found`

Use the absolute Bash path:

```bash
/bin/bash ./run_camera.sh 0
```

Do not type `hash`, and do not rely on an incorrectly configured `PATH`.

### Camera cannot be opened

Check the available camera devices and try another index:

```bash
ls -l /dev/video* 2>/dev/null
/bin/bash ./run_camera.sh 1
```

Close other programs that may already be using the camera.

### No detection window appears

Run the command from the Jetson graphical desktop terminal. Confirm that `DISPLAY` is set:

```bash
echo "$DISPLAY"
export DISPLAY=:0
```

For an SSH-only session, use the headless command in Section 5.

### Python import failure

Check which Python executable is being used and verify the imports:

```bash
which python3
python3 -c "import cv2, torch, ultralytics; print('Imports OK')"
```

### ROS 2 or `colcon` is unavailable

```bash
source /opt/ros/humble/setup.bash
which ros2
which colcon
```

The ROS 2 command must be run on a system where ROS 2 Humble and the Python package `rclpy` are installed.
