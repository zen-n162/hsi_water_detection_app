from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from hsi_water_detection_app.data.metadata import HSICube
from hsi_water_detection_app.data.roi import ROI


class BaseHSIAdapter(ABC):
    sensor_name: str = "unknown"

    @abstractmethod
    def can_handle(self, path: Path, meta_hint: dict[str, Any]) -> bool:
        ...

    @abstractmethod
    def load(self, path: Path, roi: Optional[ROI] = None) -> HSICube:
        ...
