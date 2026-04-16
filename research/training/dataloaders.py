from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_MAP = {
    "dry": 0.0,
    "wet": 1.0,
    "uncertain": 0.5,
}


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    return value


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    value = _normalize_scalar(value)
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    value = _normalize_scalar(value)
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_sidecar_json(path: Path) -> dict[str, Any]:
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to read sidecar json: {json_path} ({e!r})")
        return {}


def _extract_metadata(path: Path, data: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}

    # まず json sidecar を読む
    meta.update(_load_sidecar_json(path))

    # 次に npz 内キーがあれば優先して上書き
    if hasattr(data, "files"):
        for key in [
            "split",
            "sample_id",
            "scene_id",
            "input_path",
            "sensor",
            "row_start",
            "row_stop",
            "col_start",
            "col_stop",
            "notes",
            "source",
            "block_row",
            "block_col",
            "block_id",
            "split_policy",
        ]:
            if key in data.files:
                meta[key] = _normalize_scalar(data[key])

    return meta


class WetnessPatchDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        split: str | None = None,
        expected_hw: tuple[int, int] = (64, 64),
        allow_uncertain: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.split = split
        self.expected_hw = expected_hw
        self.allow_uncertain = allow_uncertain
        self.records: list[dict[str, Any]] = []

        if not self.input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {self.input_dir}")

        for path in sorted(self.input_dir.glob("*.npz")):
            try:
                with np.load(path, allow_pickle=True) as data:
                    if "cube" not in data.files:
                        print(f"[WARN] skip dataset file without cube: {path}")
                        continue
                    if "label" not in data.files:
                        print(f"[WARN] skip dataset file without label: {path}")
                        continue
                    if "confidence" not in data.files:
                        print(f"[WARN] skip dataset file without confidence: {path}")
                        continue

                    cube = data["cube"]
                    if cube.ndim != 3:
                        print(f"[WARN] skip dataset file due to invalid cube ndim: {path} -> {cube.shape}")
                        continue
                    if cube.shape[1:] != expected_hw:
                        print(f"[WARN] skip dataset file due to shape mismatch: {path} -> {cube.shape}")
                        continue

                    label_raw = _safe_str(data["label"]).strip().lower()
                    if label_raw not in LABEL_MAP:
                        print(f"[WARN] skip dataset file due to unknown label: {path} -> {label_raw}")
                        continue
                    if (not allow_uncertain) and label_raw == "uncertain":
                        continue

                    confidence = _safe_float(data["confidence"], default=1.0)

                    meta = _extract_metadata(path, data)
                    split_value = _safe_str(meta.get("split"), default="").strip()

                    if split is not None and split_value != split:
                        continue

                    record = {
                        "path": path,
                        "split": split_value,
                        "label_raw": label_raw,
                        "confidence": confidence,
                        "meta": meta,
                    }
                    self.records.append(record)

            except Exception as e:
                print(f"[WARN] skip unreadable dataset file: {path} ({e!r})")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path: Path = record["path"]

        with np.load(path, allow_pickle=True) as data:
            cube = data["cube"].astype(np.float32)

        label_raw = record["label_raw"]
        y = np.float32(LABEL_MAP.get(label_raw, 0.5))
        confidence = np.float32(record["confidence"])
        meta = record["meta"]

        return {
            "cube": torch.from_numpy(cube),
            "label": torch.tensor(y, dtype=torch.float32),
            "confidence": torch.tensor(confidence, dtype=torch.float32),
            "split": record["split"],
            "file": path.name,
            "path": str(path),
            "sample_id": _safe_str(meta.get("sample_id"), default=path.stem),
            "scene_id": _safe_str(meta.get("scene_id"), default=""),
            "input_path": _safe_str(meta.get("input_path"), default=""),
            "sensor": _safe_str(meta.get("sensor"), default=""),
            "row_start": _safe_float(meta.get("row_start"), default=-1),
            "row_stop": _safe_float(meta.get("row_stop"), default=-1),
            "col_start": _safe_float(meta.get("col_start"), default=-1),
            "col_stop": _safe_float(meta.get("col_stop"), default=-1),
            "notes": _safe_str(meta.get("notes"), default=""),
            "source": _safe_str(meta.get("source"), default=""),
            "block_row": _safe_float(meta.get("block_row"), default=-1),
            "block_col": _safe_float(meta.get("block_col"), default=-1),
            "block_id": _safe_str(meta.get("block_id"), default=""),
            "split_policy": _safe_str(meta.get("split_policy"), default=""),
            "confidence_tier": _safe_float(meta.get("confidence_tier"), default=0.0),
            "valid_ratio": _safe_float(meta.get("valid_ratio"), default=0.0),
            "wet_ratio_valid": _safe_float(meta.get("wet_ratio_valid"), default=0.0),
            "label_name": label_raw,
        }
