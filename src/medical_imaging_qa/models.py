from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass(frozen=True)
class FileProvenance:
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str | None = None


@dataclass(frozen=True)
class ImageSummary:
    path: str
    shape: list[int]
    dtype: str
    zooms: list[float]
    axcodes: list[str]
    affine: list[list[float]]
    affine_determinant: float
    qform_code: int
    sform_code: int
    intensity: dict[str, float | int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelStat:
    label: int
    voxel_count: int
    volume_mm3: float
    volume_ml: float
    centroid_voxel: list[float]
    centroid_world_mm: list[float]
    bbox_min_voxel: list[int]
    bbox_max_voxel: list[int]
    component_count: int
    largest_component_voxels: int
    largest_component_fraction: float
    touches_boundary: bool


@dataclass
class CaseReport:
    case_id: str
    status: str
    image: ImageSummary | None
    mask: ImageSummary | None
    findings: list[Finding] = field(default_factory=list)
    labels: list[LabelStat] = field(default_factory=list)
    provenance: dict[str, FileProvenance] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "image": asdict(self.image) if self.image is not None else None,
            "mask": asdict(self.mask) if self.mask is not None else None,
            "findings": [item.to_dict() for item in self.findings],
            "labels": [asdict(item) for item in self.labels],
            "provenance": {key: asdict(value) for key, value in self.provenance.items()},
        }


@dataclass
class BatchReport:
    cases: list[CaseReport]

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def error_cases(self) -> int:
        return sum(case.status == "error" for case in self.cases)

    @property
    def warning_cases(self) -> int:
        return sum(case.status == "warning" for case in self.cases)

    @property
    def pass_cases(self) -> int:
        return sum(case.status == "pass" for case in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_cases": self.total_cases,
                "pass_cases": self.pass_cases,
                "warning_cases": self.warning_cases,
                "error_cases": self.error_cases,
            },
            "cases": [case.to_dict() for case in self.cases],
        }


def status_from_findings(findings: list[Finding]) -> str:
    if any(item.severity == Severity.ERROR for item in findings):
        return "error"
    if any(item.severity == Severity.WARNING for item in findings):
        return "warning"
    return "pass"
