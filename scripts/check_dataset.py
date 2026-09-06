#!/usr/bin/env python3
"""Validate YOLO detection labels before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="检查自采 YOLO 数据集")
    parser.add_argument("--data", default="data/person1.yaml")
    args = parser.parse_args()

    yaml_path = Path(args.data).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    configured_root = Path(config["path"]).expanduser()
    if configured_root.is_absolute():
        dataset_root = configured_root
    else:
        candidates = (
            (yaml_path.parent.parent / configured_root).resolve(),
            (Path.cwd() / configured_root).resolve(),
            (yaml_path.parent / configured_root).resolve(),
        )
        dataset_root = next((path for path in candidates if path.exists()), candidates[0])
    names = config["names"]
    valid_ids = {int(value) for value in names.keys()} if isinstance(names, dict) else set(range(len(names)))

    summary: dict[str, object] = {"dataset_root": str(dataset_root), "splits": {}, "errors": []}
    class_counts: Counter[int] = Counter()
    errors: list[str] = summary["errors"]  # type: ignore[assignment]

    for split in ("train", "val", "test"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        split_instances = 0
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"缺少标注：{label_path}")
                continue
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_number} 应为5列，实际为{len(parts)}列")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number} 包含非数字内容")
                    continue
                if class_id not in valid_ids:
                    errors.append(f"{label_path}:{line_number} 类别编号越界：{class_id}")
                if any(value < 0.0 or value > 1.0 for value in coords):
                    errors.append(f"{label_path}:{line_number} 坐标不在0到1之间")
                if coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"{label_path}:{line_number} 宽高必须大于0")
                class_counts[class_id] += 1
                split_instances += 1
        summary["splits"][split] = {  # type: ignore[index]
            "images": len(images), "instances": split_instances,
        }

    summary["class_instances"] = {
        str(names[class_id] if isinstance(names, dict) else names[class_id]): count
        for class_id, count in sorted(class_counts.items())
        if class_id in valid_ids
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"数据集检查失败，共发现 {len(errors)} 个问题。")


if __name__ == "__main__":
    main()
