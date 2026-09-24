from __future__ import annotations

import numpy as np

from .models import Finding, Severity
from .rules import QARules


def voxel_volume_mm3(affine: np.ndarray) -> float:
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Affine must have shape (4, 4).")
    return float(abs(np.linalg.det(matrix[:3, :3])))


def affine_determinant(affine: np.ndarray) -> float:
    return float(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3]))


def validate_affine(affine: np.ndarray) -> list[Finding]:
    matrix = np.asarray(affine, dtype=float)
    findings: list[Finding] = []

    if matrix.shape != (4, 4):
        return [
            Finding(
                code="geometry.invalid_affine_shape",
                severity=Severity.ERROR,
                message="Affine matrix must have shape (4, 4).",
                context={"shape": list(matrix.shape)},
            )
        ]

    if not np.isfinite(matrix).all():
        findings.append(
            Finding(
                code="geometry.nonfinite_affine",
                severity=Severity.ERROR,
                message="Affine matrix contains non-finite values.",
            )
        )
        return findings

    determinant = affine_determinant(matrix)
    if abs(determinant) < 1e-12:
        findings.append(
            Finding(
                code="geometry.singular_affine",
                severity=Severity.ERROR,
                message="Affine matrix is singular or near-singular.",
                context={"determinant": determinant},
            )
        )

    return findings


def compare_geometry(
    shape_a: tuple[int, ...],
    affine_a: np.ndarray,
    zooms_a: tuple[float, ...],
    axcodes_a: tuple[str, ...],
    shape_b: tuple[int, ...],
    affine_b: np.ndarray,
    zooms_b: tuple[float, ...],
    axcodes_b: tuple[str, ...],
    rules: QARules,
) -> list[Finding]:
    findings: list[Finding] = []

    if tuple(shape_a) != tuple(shape_b):
        findings.append(
            Finding(
                code="pair.shape_mismatch",
                severity=Severity.ERROR,
                message="Image and mask shapes do not match.",
                context={"image_shape": list(shape_a), "mask_shape": list(shape_b)},
            )
        )

    relevant_dims = min(len(zooms_a), len(zooms_b), 3)
    spacing_a = np.asarray(zooms_a[:relevant_dims], dtype=float)
    spacing_b = np.asarray(zooms_b[:relevant_dims], dtype=float)
    if spacing_a.shape != spacing_b.shape or not np.allclose(
        spacing_a, spacing_b, atol=rules.spacing_atol, rtol=0.0
    ):
        findings.append(
            Finding(
                code="pair.spacing_mismatch",
                severity=Severity.ERROR,
                message="Image and mask voxel spacings do not match.",
                context={
                    "image_zooms": spacing_a.tolist(),
                    "mask_zooms": spacing_b.tolist(),
                    "atol": rules.spacing_atol,
                },
            )
        )

    matrix_a = np.asarray(affine_a, dtype=float)
    matrix_b = np.asarray(affine_b, dtype=float)
    if matrix_a.shape == (4, 4) and matrix_b.shape == (4, 4):
        if not np.allclose(matrix_a, matrix_b, atol=rules.affine_atol, rtol=0.0):
            max_delta = float(np.max(np.abs(matrix_a - matrix_b)))
            findings.append(
                Finding(
                    code="pair.affine_mismatch",
                    severity=Severity.ERROR,
                    message="Image and mask affine matrices do not match.",
                    context={"max_abs_delta": max_delta, "atol": rules.affine_atol},
                )
            )

    if tuple(axcodes_a) != tuple(axcodes_b):
        findings.append(
            Finding(
                code="pair.orientation_mismatch",
                severity=Severity.ERROR,
                message="Image and mask axis orientations do not match.",
                context={"image_axcodes": list(axcodes_a), "mask_axcodes": list(axcodes_b)},
            )
        )

    return findings


def voxel_to_world(affine: np.ndarray, voxel_xyz: list[float]) -> list[float]:
    if len(voxel_xyz) != 3:
        raise ValueError("voxel_xyz must contain exactly three coordinates.")
    homogeneous = np.array([*voxel_xyz, 1.0], dtype=float)
    world = np.asarray(affine, dtype=float) @ homogeneous
    return [float(value) for value in world[:3]]
