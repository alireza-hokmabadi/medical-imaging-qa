import numpy as np
import pytest

from medical_imaging_qa.rules import QARules
from medical_imaging_qa.segmentation import label_statistics, validate_labelmap


def test_label_statistics_volume_centroid_and_components():
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:4, 3:5, 4:6] = 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    stats, findings = label_statistics(
        mask,
        affine,
        QARules(warn_boundary_touch=False),
    )

    assert len(stats) == 1
    item = stats[0]
    assert item.label == 1
    assert item.voxel_count == 8
    assert item.volume_mm3 == pytest.approx(64.0)
    assert item.volume_ml == pytest.approx(0.064)
    assert item.component_count == 1
    assert item.centroid_voxel == [2.5, 3.5, 4.5]
    assert findings == []


def test_fragmented_label_warns():
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:6, 3:6, 3:6] = 1
    mask[9, 9, 9] = 1
    _, findings = label_statistics(mask, np.eye(4), QARules(min_component_voxels=2))
    codes = {item.code for item in findings}
    assert "mask.multiple_components" in codes
    assert "mask.small_components" in codes


def test_boundary_touch_warns():
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[0:2, 2:4, 2:4] = 3
    _, findings = label_statistics(mask, np.eye(4), QARules())
    assert "mask.touches_boundary" in {item.code for item in findings}


def test_noninteger_mask_errors():
    mask = np.zeros((4, 4, 4), dtype=float)
    mask[1, 1, 1] = 1.5
    findings = validate_labelmap(mask, QARules())
    assert "mask.noninteger_labels" in {item.code for item in findings}


def test_expected_labels_are_checked():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    findings = validate_labelmap(mask, QARules(), expected_labels={1, 2})
    missing = [item for item in findings if item.code == "mask.missing_expected_labels"]
    assert missing[0].context["labels"] == [2]
