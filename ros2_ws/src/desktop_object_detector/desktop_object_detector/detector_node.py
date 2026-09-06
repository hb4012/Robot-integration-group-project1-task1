#!/usr/bin/env python3
"""ROS 2 node that publishes YOLO detections as JSON and FPS as Float32."""

from __future__ import annotations

import json
import time
from collections import deque

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from ultralytics import YOLO


TARGET_NAMES = ("bottle", "mouse", "laptop")


def model_name(names: dict | list, class_id: int) -> str:
    return str(names[class_id] if isinstance(names, list) else names.get(class_id, class_id))


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

        model_path = str(self.get_parameter("model_path").value)
        source_text = str(self.get_parameter("source").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.iou = float(self.get_parameter("iou").value)
        self.device = str(self.get_parameter("device").value)
        self.display = bool(self.get_parameter("display").value)

        self.model = YOLO(model_path)
        self.target_ids = [
            idx for idx in range(len(self.model.names))
            if model_name(self.model.names, idx) in TARGET_NAMES
        ]
        present = {model_name(self.model.names, idx) for idx in self.target_ids}
        missing = sorted(set(TARGET_NAMES) - present)
        if missing:
            raise RuntimeError(f"模型缺少类别：{', '.join(missing)}")

        source: int | str = int(source_text) if source_text.isdigit() else source_text
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开输入源：{source_text}")

        self.detection_pub = self.create_publisher(String, "/desktop_detector/detections", 10)
        self.fps_pub = self.create_publisher(Float32, "/desktop_detector/fps", 10)
        self.frame_id = 0
        self.recent_times: deque[float] = deque(maxlen=31)
        self.timer = self.create_timer(0.001, self.process_frame)
        self.get_logger().info(f"已加载模型：{model_path}")

    def process_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("无法读取下一帧，停止节点。")
            rclpy.shutdown()
            return

        self.frame_id += 1
        self.recent_times.append(time.perf_counter())
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            classes=self.target_ids,
            verbose=False,
        )[0]
        fps = 0.0
        if len(self.recent_times) > 1:
            fps = (len(self.recent_times) - 1) / (self.recent_times[-1] - self.recent_times[0])

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
            "fps": round(fps, 3),
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
                rclpy.shutdown()

    def destroy_node(self) -> bool:
        self.capture.release()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
