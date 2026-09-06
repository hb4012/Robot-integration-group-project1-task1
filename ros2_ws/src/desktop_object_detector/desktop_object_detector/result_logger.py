#!/usr/bin/env python3
"""Save /desktop_detector/detections messages to a JSONL file."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ResultLogger(Node):
    def __init__(self) -> None:
        super().__init__("desktop_detection_logger")
        self.declare_parameter("output_path", "results/ros_detections.jsonl")
        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_handle = self.output_path.open("a", encoding="utf-8")
        self.count = 0
        self.frames_with_detections = 0
        self.class_counts: Counter[str] = Counter()
        self.subscription = self.create_subscription(
            String, "/desktop_detector/detections", self.on_detection, 10
        )
        self.get_logger().info(f"检测结果将保存到：{self.output_path.resolve()}")

    def on_detection(self, message: String) -> None:
        self.output_handle.write(message.data + "\n")
        self.output_handle.flush()
        self.count += 1
        try:
            payload = json.loads(message.data)
            detections = payload.get("detections", [])
            self.frames_with_detections += int(bool(detections))
            for detection in detections:
                self.class_counts[str(detection.get("class_name", "unknown"))] += 1
        except (TypeError, ValueError):
            self.get_logger().warning("收到无法解析的检测消息")
        if self.count % 30 == 0:
            self.get_logger().info(f"已保存 {self.count} 条检测消息")

    def destroy_node(self) -> bool:
        self.output_handle.close()
        summary = {
            "messages": self.count,
            "frames_with_detections": self.frames_with_detections,
            "detection_counts": dict(self.class_counts),
            "source_topic": "/desktop_detector/detections",
        }
        self.output_path.with_name("summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ResultLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
