#!/usr/bin/env python3
"""Export the trained checkpoint to a Jetson-specific TensorRT FP16 engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from runtime_utils import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="在目标Jetson上导出TensorRT FP16模型")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models/best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=float, default=2.0, help="单位GiB")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型：{model_path}")
    exported = YOLO(str(model_path)).export(
        format="engine",
        imgsz=args.imgsz,
        half=True,
        device=0,
        batch=1,
        workspace=args.workspace,
        dynamic=False,
        simplify=True,
    )
    print(f"TensorRT模型已生成：{exported}")


if __name__ == "__main__":
    main()
