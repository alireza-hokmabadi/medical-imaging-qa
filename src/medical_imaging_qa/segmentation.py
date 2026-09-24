from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import voxel_to_world, voxel_volume_mm3
from .models import Finding, LabelStat, Severity
from .rules import QARules


def validate_labelmap(
    data: np.ndarray,
    rules: QARules,
    expected_labels: set[int] | None = None,
    background: int = 0,
) -> list[Finding]:
    array = np.asarray(data)
    findings: list[Finding] = []

    if array.ndim != 3:
        findings.append(
            Finding(
                code="mask.unsupported_dimensions",
                severity=Severity.ERROR,
                message="Label-map QA currently expects a 3D mask.",
                context={"ndim": int(array.ndim), "shape": list(array.shape)},
            )
        )
        return findings

    if array.size == 0:
        return [
            Finding(
                code="mask.empty_array",
                severity=Severity.ERROR,
                message="Mask contains no voxels.",
            )
        ]

    finite = np.isfinite(array)
    if not finite.all():
        findings.append(
            Finding(
                code="mask.nonfinite_values",
                severity=Severity.ERROR,
                message="Mask contains NaN or infinite values.",
                context={"nonfinite_voxels": int(np.count_nonzero(~finite))},
            )
        )
        return findings

    rounded = np.rint(array)
    max_fractional = float(np.max(np.abs(array - rounded)))
    if max_fractional > rules.integer_tolerance:
        findings.append(
            Finding(
                code="mask.noninteger_labels",
                severity=Severity.ERROR,
                message="Mask contains non-integer label values.",
                context={
                    "max_fractional_distance": max_fractional,
                    "tolerance": rules.integer_tolerance,
                },
            )
        )
        return findings

    labels = {int(value) for value in np.unique(rounded)}
    foreground = labels - {background}
    if not foreground:
        findings.append(
            Finding(
                code="mask.no_foreground",
                severity=Severity.ERROR,
                message="Mask contains no foreground labels.",
                context={"background": background},
            )
        )

    negative = sorted(label for label in labels if label < 0)
    if negative:
        findings.append(
            Finding(
                code="mask.negative_labels",
                severity=Severity.WARNING,
                message="Mask contains negative label values.",
                context={"labels": negative},
            )
        )

    if expected_labels is not None:
        missing = sorted(expected_labels - foreground)
        unexpected = sorted(foreground - expected_labels)
        if missing:
            findings.append(
                Finding(
                    code="mask.missing_expected_labels",
                    severity=Severity.WARNING,
                    message="One or more expected foreground labels are missing.",
                    context={"labels": missing},
                )
            )
        if unexpected:
            findings.append(
                Finding(
                    code="mask.unexpected_labels",
                    severity=Severity.WARNING,
                    message="Mask contains labels outside the expected set.",
                    context={"labels": unexpected},
                )
            )

    return findings


def _touches_boundary(binary: np.ndarray) -> bool:
    return bool(
        np.any(binary[0, :, :])
        or np.any(binary[-1, :, :])
        or np.any(binary[:, 0, :])
        or np.any(binary[:, -1, :])
        or np.any(binary[:, :, 0])
        or np.any(binary[:, :, -1])
    )


def label_statistics(
    data: np.ndarray,
    affine: np.ndarray,
    rules: QARules,
    background: int = 0,
) -> tuple[list[LabelStat], list[Finding]]:
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError("label_statistics expects a 3D label map.")
    if not np.isfinite(array).all():
        raise ValueError("label_statistics requires finite label values.")

    labels = np.rint(array).astype(np.int64)
    unique_labels = [int(value) for value in np.unique(labels) if int(value) != background]
    voxel_volume = voxel_volume_mm3(affine)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

    stats: list[LabelStat] = []
    findings: list[Finding] = []

    for label_value in unique_labels:
        binary = labels == label_value
        coordinates = np.argwhere(binary)
        voxel_count = int(coordinates.shape[0])
        centroid_voxel = [float(value) for value in coordinates.mean(axis=0)]
        bbox_min = [int(value) for value in coordinates.min(axis=0)]
        bbox_max = [int(value) for value in coordinates.max(axis=0)]

        component_map, component_count = ndimage.label(binary, structure=structure)
        component_sizes = np.bincount(component_map.ravel())[1:]
        largest_component = int(component_sizes.max()) if component_sizes.size else 0
        largest_fraction = float(largest_component / voxel_count) if voxel_count else 0.0
        touches_boundary = _touches_boundary(binary)

        stats.append(
            LabelStat(
                label=label_value,
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count * voxel_volume),
                volume_ml=float(voxel_count * voxel_volume / 1000.0),
                centroid_voxel=centroid_voxel,
                centroid_world_mm=voxel_to_world(affine, centroid_voxel),
                bbox_min_voxel=bbox_min,
                bbox_max_voxel=bbox_max,
                component_count=int(component_count),
                largest_component_voxels=largest_component,
                largest_component_fraction=largest_fraction,
                touches_boundary=touches_boundary,
            )
        )

        if rules.warn_multiple_components and component_count > 1:
            findings.append(
                Finding(
                    code="mask.multiple_components",
                    severity=Severity.WARNING,
                    message=f"Label {label_value} contains multiple connected components.",
                    context={
                        "label": label_value,
                        "component_count": int(component_count),
                        "largest_component_fraction": largest_fraction,
                    },
                )
            )

        small_components = int(np.count_nonzero(component_sizes < rules.min_component_voxels))
        if small_components:
            findings.append(
                Finding(
                    code="mask.small_components",
                    severity=Severity.WARNING,
                    message=f"Label {label_value} contains small connected components.",
                    context={
                        "label": label_value,
                        "small_component_count": small_components,
                        "threshold_voxels": rules.min_component_voxels,
                    },
                )
            )

        if rules.warn_boundary_touch and touches_boundary:
            findings.append(
                Finding(
                    code="mask.touches_boundary",
                    severity=Severity.WARNING,
                    message=f"Label {label_value} touches the image boundary.",
                    context={"label": label_value},
                )
            )

    return stats, findings
