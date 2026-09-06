#!/usr/bin/env python3
"""Shared paths, classes, input parsing, and device selection."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = ("bottle", "mouse", "laptop")


def class_name(names: Mapping | Sequence, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    return str(names[class_id])


def class_ids(names: Mapping | Sequence) -> list[int]:
    ids = sorted(int(key) for key in names) if isinstance(names, Mapping) else list(range(len(names)))
    return [class_id for class_id in ids if class_name(names, class_id) in TARGET_NAMES]


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


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
