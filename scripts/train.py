#!/usr/bin/env python3
"""Fine-tune YOLO11n on the three-class desktop-object dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练桌面物体检测模型")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练模型名称或本地权重路径")
    parser.add_argument("--data", default=str(ROOT / "data/person1.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps", help="Mac 用 mps；NVIDIA GPU 用 0；CPU 用 cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="person1_3classes_yolo11n")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到数据配置：{data_path}")

    # Ultralytics resolves a relative dataset `path` from the current working
    # directory. Always train from the project root so the portable YAML works
    # whether this script is launched from the Mac project or a copied Jetson.
    os.chdir(ROOT)
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(ROOT / "results/training_runs"),
        name=args.name,
        seed=42,
        plots=True,
    )
    best = ROOT / "results/training_runs" / args.name / "weights/best.pt"
    print(f"\n训练结束。最佳权重通常位于：{best}")


if __name__ == "__main__":
    main()
