from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class AnnotationPayload(BaseModel):
    sample_id: str
    scene_id: str
    input_path: str
    sensor: str
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    label: str
    confidence: float
    source: str = "manual_ui"
    notes: str = ""


@router.post("/annotations/roi")
def save_roi_annotation(payload: AnnotationPayload):
    outdir = Path("annotations/roi_labels")
    outdir.mkdir(parents=True, exist_ok=True)

    path = outdir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{payload.sample_id}.json"
    path.write_text(json.dumps(payload.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    return {"ok": True, "path": str(path)}
