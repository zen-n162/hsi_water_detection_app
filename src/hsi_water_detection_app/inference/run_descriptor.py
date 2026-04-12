from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RunDescriptor:
    sensor: str
    model_type: str
    input_path: str
    output_dir: str
    spat_checkpoint: Optional[str] = None
    spec_checkpoint: Optional[str] = None
    row_start: Optional[int] = None
    row_stop: Optional[int] = None
    col_start: Optional[int] = None
    col_stop: Optional[int] = None

    def to_dict(self):
        return asdict(self)
