import numpy as np

from medical_imaging_qa.intensity import intensity_statistics, validate_intensity
from medical_imaging_qa.rules import QARules


def test_intensity_statistics():
    data = np.array([0.0, 1.0, 2.0, 3.0])
    stats = intensity_statistics(data)
    assert stats["voxel_count"] == 4
    assert stats["min"] == 0.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 1.5


def test_nonfinite_values_are_reported():
    data = np.array([0.0, np.nan, 1.0])
    findings = validate_intensity(data, QARules())
    assert {item.code for item in findings} == {"image.nonfinite_values"}


def test_constant_image_warns():
    findings = validate_intensity(np.ones((3, 3, 3)), QARules())
    assert "image.constant_intensity" in {item.code for item in findings}
