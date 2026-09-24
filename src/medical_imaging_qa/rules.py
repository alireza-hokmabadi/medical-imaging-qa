from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QARules:
    affine_atol: float = 1e-4
    spacing_atol: float = 1e-5
    integer_tolerance: float = 1e-6
    min_component_voxels: int = 8
    warn_multiple_components: bool = True
    warn_boundary_touch: bool = True
    warn_missing_qform_sform: bool = True
    max_nonfinite_fraction: float = 0.0

    @classmethod
    def from_json(cls, path: str | Path) -> QARules:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("QA rules JSON must contain an object at the top level.")
        unknown = sorted(set(payload) - set(asdict(cls()).keys()))
        if unknown:
            raise ValueError(f"Unknown QA rule(s): {', '.join(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
