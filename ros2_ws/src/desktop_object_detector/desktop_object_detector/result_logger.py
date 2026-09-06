#!/usr/bin/env python3
"""Save /desktop_detector/detections messages to a JSONL file."""

from __future__ import annotations

import json
import math
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
        self.output_handle = self.output_path.open("a", encoding="utf-8", buffering=1)
        self.count = 0
        self.frames_with_detections = 0
        self.total_detections = 0
        self.malformed_messages = 0
        self.class_counts: Counter[str] = Counter()
        self.fps_count = 0
        self.fps_sum = 0.0
        self.fps_min: float | None = None
        self.fps_max: float | None = None
        self.first_stamp_nanoseconds: int | None = None
        self.last_stamp_nanoseconds: int | None = None
        self._closed = False
        self.subscription = self.create_subscription(
            String, "/desktop_detector/detections", self.on_detection, 10
        )
        self.get_logger().info(f"Detection results will be saved to {self.output_path.resolve()}")

    def on_detection(self, message: String) -> None:
        self.output_handle.write(message.data + "\n")
        self.count += 1
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise TypeError("the JSON root must be an object")
            detections = payload.get("detections", [])
            if not isinstance(detections, list):
                raise TypeError("detections must be a list")
            self.frames_with_detections += int(bool(detections))
            self.total_detections += len(detections)
            for detection in detections:
                if isinstance(detection, dict):
                    self.class_counts[str(detection.get("class_name", "unknown"))] += 1

            fps = float(payload.get("fps", 0.0))
            if math.isfinite(fps) and fps > 0.0:
                self.fps_count += 1
                self.fps_sum += fps
                self.fps_min = fps if self.fps_min is None else min(self.fps_min, fps)
                self.fps_max = fps if self.fps_max is None else max(self.fps_max, fps)

            stamp = int(payload["stamp_nanoseconds"])
            if self.first_stamp_nanoseconds is None:
                self.first_stamp_nanoseconds = stamp
            self.last_stamp_nanoseconds = stamp
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.malformed_messages += 1
            self.get_logger().warning(f"Unable to parse a detection message: {exc}")
        if self.count % 30 == 0:
            self.get_logger().info(
                f"Saved {self.count} messages containing {self.total_detections} detections"
            )

    def write_summary(self) -> None:
        duration_seconds = None
        if self.first_stamp_nanoseconds is not None and self.last_stamp_nanoseconds is not None:
            duration_seconds = max(
                0.0,
                (self.last_stamp_nanoseconds - self.first_stamp_nanoseconds) / 1_000_000_000.0,
            )
        average_fps = self.fps_sum / self.fps_count if self.fps_count else 0.0
        summary = {
            "messages": self.count,
            "frames_with_detections": self.frames_with_detections,
            "total_detections": self.total_detections,
            "malformed_messages": self.malformed_messages,
            "duration_seconds": None if duration_seconds is None else round(duration_seconds, 3),
            "fps": {
                "samples": self.fps_count,
                "average": round(average_fps, 3),
                "minimum": None if self.fps_min is None else round(self.fps_min, 3),
                "maximum": None if self.fps_max is None else round(self.fps_max, 3),
            },
            "detection_counts": dict(sorted(self.class_counts.items())),
            "source_topic": "/desktop_detector/detections",
        }
        summary_path = self.output_path.with_name("summary.json")
        temporary_path = summary_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(summary_path)

    def destroy_node(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        self.output_handle.close()
        self.write_summary()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: ResultLogger | None = None
    try:
        node = ResultLogger()
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
