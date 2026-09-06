#!/usr/bin/env python3
"""Evaluate a fixed 20-case acceptance manifest and save typical errors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def get_name(names: dict | list, class_id: int) -> str:
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
    parser = argparse.ArgumentParser(description="执行固定20例验收测试")
    parser.add_argument("--model", required=True, help="best.pt 或 TensorRT engine")
    parser.add_argument("--manifest", required=True, help="CSV：case_id、image和3个类别的真实数量")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--output", default=str(ROOT / "results/acceptance_20"))
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output = Path(args.output)
    annotated_dir = output / "annotated"
    error_dir = output / "error_cases"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        cases = list(csv.DictReader(handle))
    if len(cases) != 20:
        raise ValueError(f"验收清单必须正好有20行，目前有{len(cases)}行。")

    model = YOLO(args.model)
    target_names = ("bottle", "mouse", "laptop")
    class_ids = {
        get_name(model.names, idx): idx
        for idx in range(len(model.names))
        if get_name(model.names, idx) in target_names
    }
    missing = set(target_names) - set(class_ids)
    if missing:
        raise RuntimeError(f"模型缺少类别：{sorted(missing)}")

    rows: list[dict[str, object]] = []
    correct_cases = 0
    for case in cases:
        case_id = case["case_id"]
        image_path = Path(case["image"])
        if not image_path.is_absolute():
            image_path = (manifest.parent / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        expected = {name: int(case.get(name, 0) or 0) for name in target_names}
        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=resolve_device(args.device),
            classes=list(class_ids.values()),
            verbose=False,
        )[0]
        predicted: Counter[str] = Counter()
        if result.boxes is not None:
            for box in result.boxes:
                predicted[get_name(result.names, int(box.cls[0].item()))] += 1
        predicted_dict = {name: predicted.get(name, 0) for name in target_names}
        is_correct = predicted_dict == expected
        correct_cases += int(is_correct)

        annotated = result.plot()
        destination = annotated_dir / f"case_{case_id}.jpg"
        cv2.imwrite(str(destination), annotated)
        if not is_correct:
            cv2.imwrite(str(error_dir / f"case_{case_id}.jpg"), annotated)
        rows.append({
            "case_id": case_id,
            "image": str(image_path),
            "expected": json.dumps(expected, ensure_ascii=False),
            "predicted": json.dumps(predicted_dict, ensure_ascii=False),
            "correct": is_correct,
        })

    output.mkdir(parents=True, exist_ok=True)
    with (output / "test_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_cases": len(cases),
        "correct_cases": correct_cases,
        "accuracy_percent": round(correct_cases / len(cases) * 100, 2),
        "requirement_met": correct_cases >= 16,
        "internal_target_met": correct_cases >= 18,
        "confidence_threshold": args.conf,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
