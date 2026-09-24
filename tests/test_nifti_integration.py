import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from medical_imaging_qa.api import validate_pair
from medical_imaging_qa.synthetic import create_demo_dataset


def test_validate_pair_clean(tmp_path):
    image = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 1
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    report = validate_pair(image_path, mask_path, expected_labels={1})
    assert report.status in {"pass", "warning"}
    assert not any(item.code.startswith("pair.") for item in report.findings)
    assert report.labels[0].label == 1


def test_synthetic_demo_contains_expected_cases(tmp_path):
    manifest = create_demo_dataset(tmp_path)
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "clean" in text
    assert "fragmented" in text
    assert "geometry_mismatch" in text
