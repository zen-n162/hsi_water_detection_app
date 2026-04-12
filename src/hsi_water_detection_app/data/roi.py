from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ROI:
    mode: str  # "pixel" | "bounds"
    row_start: Optional[int] = None
    row_stop: Optional[int] = None
    col_start: Optional[int] = None
    col_stop: Optional[int] = None
    xmin: Optional[float] = None
    ymin: Optional[float] = None
    xmax: Optional[float] = None
    ymax: Optional[float] = None
