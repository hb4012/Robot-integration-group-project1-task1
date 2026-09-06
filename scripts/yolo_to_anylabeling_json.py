#!/usr/bin/env python3
"""Convert this project's YOLO detection labels to AnyLabeling JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


CLASS_NAMES = [
    "bottle",
    "cup",
    "book",
    "mouse",
    "keyboard",
    "laptop",
    "cell phone",
    "scissors",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_box_to_rectangle(
    values: list[float], image_width: int, image_height: int
) -> list[list[float]]:
    """Convert normalized YOLO xywh coordinates to two rectangle corners."""
    x_center, y_center, box_width, box_height = values
    x1 = max(0.0, (x_center - box_width / 2) * image_width)
    y1 = max(0.0, (y_center - box_height / 2) * image_height)
    x2 = min(float(image_width), (x_center + box_width / 2) * image_width)
    y2 = min(float(image_height), (y_center + box_height / 2) * image_height)
    return [[round(x1, 3), round(y1, 3)], [round(x2, 3), round(y2, 3)]]


def convert_split(dataset_dir: Path, split: str) -> tuple[int, int]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    output_dir = dataset_dir / "annotations_json" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    box_count = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing YOLO label: {label_path}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        shapes = []
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO row in {label_path}:{line_number}")

            class_id = int(fields[0])
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(
                    f"Unknown class id {class_id} in {label_path}:{line_number}"
                )
            coordinates = [float(value) for value in fields[1:]]
            shapes.append(
                {
                    "label": CLASS_NAMES[class_id],
                    "text": "",
                    "points": yolo_box_to_rectangle(
                        coordinates, image_width, image_height
                    ),
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )

        output_path = output_dir / f"{image_path.stem}.json"
        relative_image_path = os.path.relpath(image_path, output_dir)
        annotation = {
            "version": "0.4.36",
            "flags": {},
            "shapes": shapes,
            "imagePath": relative_image_path,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }
        output_path.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        box_count += len(shapes)

    return len(images), box_count


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert YOLO TXT labels to AnyLabeling JSON annotations."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_dir / "datasets" / "desktop_objects",
        help="Dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to convert (default: train val).",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset.expanduser().resolve()
    total_images = 0
    total_boxes = 0
    for split in args.splits:
        image_count, box_count = convert_split(dataset_dir, split)
        total_images += image_count
        total_boxes += box_count
        print(f"{split}: {image_count} JSON files, {box_count} boxes")

    print(f"total: {total_images} JSON files, {total_boxes} boxes")
    print(f"output: {dataset_dir / 'annotations_json'}")


if __name__ == "__main__":
    main()
