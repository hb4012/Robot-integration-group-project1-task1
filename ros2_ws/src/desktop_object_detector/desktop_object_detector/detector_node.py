#!/usr/bin/env python3
"""ROS 2 node that publishes YOLO detections as JSON and FPS as Float32."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from ultralytics import YOLO


TARGET_NAMES = ("bottle", "mouse", "laptop")
DEFAULT_DETECTIONS_TOPIC = "/desktop_detector/detections"
DEFAULT_FPS_TOPIC = "/desktop_detector/fps"


def model_name(names: Mapping | Sequence, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    return str(names[class_id])


def target_class_ids(names: Mapping | Sequence) -> list[int]:
    if isinstance(names, Mapping):
        class_ids = sorted(int(class_id) for class_id in names)
    else:
        class_ids = list(range(len(names)))
    return [class_id for class_id in class_ids if model_name(names, class_id) in TARGET_NAMES]


def require_range(name: str, value: float, minimum: float, maximum: float) -> float:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}; received {value}")
    return value


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("desktop_object_detector")
        self.declare_parameter("model_path", "models/best.pt")
        self.declare_parameter("source", "0")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("confidence", 0.40)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "0")
        self.declare_parameter("display", True)
        self.declare_parameter("detections_topic", DEFAULT_DETECTIONS_TOPIC)
        self.declare_parameter("fps_topic", DEFAULT_FPS_TOPIC)

        model_path = str(self.get_parameter("model_path").value)
        source_text = str(self.get_parameter("source").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.confidence = require_range(
            "confidence", float(self.get_parameter("confidence").value), 0.0, 1.0
        )
        self.iou = require_range("iou", float(self.get_parameter("iou").value), 0.0, 1.0)
        self.device = str(self.get_parameter("device").value)
        self.display = bool(self.get_parameter("display").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        fps_topic = str(self.get_parameter("fps_topic").value)

        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be positive; received {self.imgsz}")
        if not model_path:
            raise ValueError("model_path cannot be empty")
        if not source_text:
            raise ValueError("source cannot be empty")
        if not self.device:
            raise ValueError("device cannot be empty")

        self.model = YOLO(model_path)
        self.target_ids = target_class_ids(self.model.names)
        present = {model_name(self.model.names, idx) for idx in self.target_ids}
        missing = sorted(set(TARGET_NAMES) - present)
        if missing:
            raise RuntimeError(f"Model is missing target classes: {', '.join(missing)}")

        source: int | str = int(source_text) if source_text.isdigit() else source_text
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open input source: {source_text}")
        if isinstance(source, int):
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.detection_pub = self.create_publisher(String, detections_topic, 10)
        self.fps_pub = self.create_publisher(Float32, fps_topic, 10)
        self.frame_id = 0
        self.recent_times: deque[float] = deque(maxlen=31)
        self._stopping = False
        self.timer = self.create_timer(0.001, self.process_frame)
        self.get_logger().info(
            f"Loaded {model_path}; source={source_text}; device={self.device}; "
            f"classes={','.join(sorted(present))}"
        )

    def process_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("Unable to read the next frame; stopping the node.")
            self.request_shutdown()
            return

        self.frame_id += 1
        started_at = time.perf_counter()
        try:
            result = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                classes=self.target_ids,
                verbose=False,
            )[0]
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            self.request_shutdown()
            return
        finished_at = time.perf_counter()
        self.recent_times.append(finished_at)
        fps = 0.0
        if len(self.recent_times) > 1:
            duration = self.recent_times[-1] - self.recent_times[0]
            if duration > 0.0:
                fps = (len(self.recent_times) - 1) / duration

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append({
                    "class_id": cls_id,
                    "class_name": model_name(result.names, cls_id),
                    "confidence": round(float(box.conf[0].item()), 5),
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                })

        payload = {
            "stamp_nanoseconds": self.get_clock().now().nanoseconds,
            "frame_id": self.frame_id,
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
            "fps": round(fps, 3),
            "processing_ms": round((finished_at - started_at) * 1000.0, 3),
            "num_detections": len(detections),
            "detections": detections,
        }
        detection_message = String()
        detection_message.data = json.dumps(payload, ensure_ascii=False)
        self.detection_pub.publish(detection_message)
        fps_message = Float32()
        fps_message.data = float(fps)
        self.fps_pub.publish(fps_message)

        if self.display:
            annotated = result.plot()
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imshow("ROS2 YOLO11 - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.request_shutdown()

    def request_shutdown(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.timer.cancel()
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        self.capture.release()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: DetectorNode | None = None
    try:
        node = DetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
