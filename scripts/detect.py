#!/usr/bin/env python3
"""Run real-time detection and save annotated video, CSV, frames, and a summary."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = ("bottle", "mouse", "laptop")


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时检测桌面物体")
    parser.add_argument("--model", default=str(ROOT / "models/best.pt"))
    parser.add_argument("--source", default="0", help="摄像头编号、视频或图片路径")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto", help="auto、mps、0或cpu")
    parser.add_argument("--output", default=str(ROOT / "results/detect"))
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示不限制")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-frames", type=int, default=0, help="每 N 帧保存一张；0 不保存")
    parser.add_argument("--headless", action="store_true", help="不弹出实时窗口")
    parser.add_argument("--all-classes", action="store_true", help="不限制为本项目的3个目标类别")
    return parser.parse_args()


def class_name(names: dict | list, class_id: int) -> str:
    return str(names[class_id] if isinstance(names, list) else names.get(class_id, class_id))


def resolve_device(requested: str) -> str:
    if requested.lower() != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    target_ids = [idx for idx in range(len(names)) if class_name(names, idx) in TARGET_NAMES]
    present = {class_name(names, idx) for idx in target_ids}
    missing = sorted(set(TARGET_NAMES) - present)
    if missing and not args.all_classes:
        raise RuntimeError(f"模型缺少类别：{', '.join(missing)}")

    device = resolve_device(args.device)

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入源：{args.source}")

    writer = None
    csv_path = output_dir / "detections.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    fields = [
        "timestamp", "frame_id", "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "inference_ms", "end_to_end_fps",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
    csv_writer.writeheader()

    frame_id = 0
    started = time.perf_counter()
    recent_times: deque[float] = deque(maxlen=31)
    class_counts: Counter[str] = Counter()
    simultaneous_saved = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            now = time.perf_counter()
            recent_times.append(now)

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                classes=None if args.all_classes else target_ids,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()
            if len(recent_times) > 1:
                fps = (len(recent_times) - 1) / (recent_times[-1] - recent_times[0])
            else:
                fps = 0.0
            inference_ms = float(result.speed.get("inference", 0.0))
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            boxes = result.boxes
            frame_classes: set[str] = set()
            if boxes is None or len(boxes) == 0:
                csv_writer.writerow({
                    "timestamp": time.time(), "frame_id": frame_id,
                    "inference_ms": f"{inference_ms:.3f}", "end_to_end_fps": f"{fps:.3f}",
                })
            else:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    name = class_name(result.names, cls_id)
                    frame_classes.add(name)
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    class_counts[name] += 1
                    csv_writer.writerow({
                        "timestamp": time.time(), "frame_id": frame_id,
                        "class_id": cls_id, "class_name": name,
                        "confidence": f"{confidence:.5f}",
                        "x1": f"{x1:.2f}", "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}", "y2": f"{y2:.2f}",
                        "inference_ms": f"{inference_ms:.3f}",
                        "end_to_end_fps": f"{fps:.3f}",
                    })

            if len(frame_classes) >= 2 and not simultaneous_saved:
                cv2.imwrite(str(output_dir / "simultaneous_two_classes.jpg"), annotated)
                simultaneous_saved = True

            if args.save_video:
                if writer is None:
                    height, width = annotated.shape[:2]
                    source_fps = cap.get(cv2.CAP_PROP_FPS)
                    source_fps = source_fps if source_fps and source_fps > 0 else 20.0
                    writer = cv2.VideoWriter(
                        str(output_dir / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )
                writer.write(annotated)

            if args.save_frames and frame_id % args.save_frames == 0:
                cv2.imwrite(str(frames_dir / f"frame_{frame_id:06d}.jpg"), annotated)

            if not args.headless:
                cv2.imshow("YOLO11 Desktop Detector - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frame_id >= args.max_frames:
                break
    finally:
        duration = max(time.perf_counter() - started, 1e-9)
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

    summary = {
        "model": args.model,
        "source": args.source,
        "frames": frame_id,
        "duration_seconds": round(duration, 3),
        "average_end_to_end_fps": round(frame_id / duration, 3),
        "detection_counts": dict(class_counts),
        "confidence_threshold": args.conf,
        "image_size": args.imgsz,
        "device": device,
        "simultaneous_two_classes_saved": simultaneous_saved,
        "jetson_5_fps_requirement_met": round(frame_id / duration, 3) >= 5.0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
