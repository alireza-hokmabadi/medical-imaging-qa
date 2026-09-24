from __future__ import annotations

import numpy as np

from .models import Finding, Severity
from .rules import QARules


def intensity_statistics(data: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(data)
    finite = np.isfinite(array)
    finite_values = array[finite]

    stats: dict[str, float | int | None] = {
        "voxel_count": int(array.size),
        "finite_voxel_count": int(finite_values.size),
        "nonfinite_voxel_count": int(array.size - finite_values.size),
        "finite_fraction": float(finite_values.size / array.size) if array.size else 0.0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "p01": None,
        "p50": None,
        "p99": None,
        "zero_fraction": None,
    }

    if finite_values.size:
        values = finite_values.astype(float, copy=False)
        stats.update(
            {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p01": float(np.percentile(values, 1)),
                "p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)),
                "zero_fraction": float(np.count_nonzero(values == 0) / values.size),
            }
        )

    return stats


def validate_intensity(data: np.ndarray, rules: QARules) -> list[Finding]:
    array = np.asarray(data)
    findings: list[Finding] = []

    if array.size == 0:
        return [
            Finding(
                code="image.empty_array",
                severity=Severity.ERROR,
                message="Image contains no voxels.",
            )
        ]

    nonfinite_fraction = float(np.count_nonzero(~np.isfinite(array)) / array.size)
    if nonfinite_fraction > rules.max_nonfinite_fraction:
        findings.append(
            Finding(
                code="image.nonfinite_values",
                severity=Severity.ERROR,
                message="Image contains non-finite intensity values.",
                context={
                    "nonfinite_fraction": nonfinite_fraction,
                    "allowed_fraction": rules.max_nonfinite_fraction,
                },
            )
        )

    finite = array[np.isfinite(array)]
    if finite.size and float(np.max(finite)) == float(np.min(finite)):
        findings.append(
            Finding(
                code="image.constant_intensity",
                severity=Severity.WARNING,
                message="All finite image voxels have the same intensity.",
                context={"value": float(finite[0])},
            )
        )

    return findings
