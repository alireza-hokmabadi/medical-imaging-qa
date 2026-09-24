import numpy as np
import pytest

from medical_imaging_qa.geometry import compare_geometry, voxel_to_world, voxel_volume_mm3
from medical_imaging_qa.rules import QARules


def test_voxel_volume_from_affine():
    affine = np.diag([1.5, 2.0, 3.0, 1.0])
    assert voxel_volume_mm3(affine) == pytest.approx(9.0)


def test_voxel_to_world():
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    assert voxel_to_world(affine, [1.0, 2.0, 3.0]) == [2.0, 6.0, 12.0]


def test_compare_geometry_detects_affine_mismatch():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = 2.0
    findings = compare_geometry(
        (10, 10, 10), a, (1.0, 1.0, 1.0), ("R", "A", "S"),
        (10, 10, 10), b, (1.0, 1.0, 1.0), ("R", "A", "S"),
        QARules(),
    )
    assert "pair.affine_mismatch" in {item.code for item in findings}


def test_compare_geometry_clean_pair_has_no_findings():
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    findings = compare_geometry(
        (10, 11, 12), affine, (1.0, 1.0, 2.0), ("R", "A", "S"),
        (10, 11, 12), affine, (1.0, 1.0, 2.0), ("R", "A", "S"),
        QARules(),
    )
    assert findings == []
