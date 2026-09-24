import hashlib

import nibabel as nib
import numpy as np
import pytest

from medical_imaging_qa.api import inspect_nifti, run_batch


def _write_pair(tmp_path, name="case_001"):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    shape = (8, 9, 10)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])

    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:6, 2:7, 3:8] = 1

    image_path = data_dir / f"{name}_image.nii.gz"
    mask_path = data_dir / f"{name}_mask.nii.gz"

    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    return image_path, mask_path


def test_inspect_image_and_mask(tmp_path):
    image_path, mask_path = _write_pair(tmp_path)

    image_report = inspect_nifti(image_path, role="image")
    assert image_report.image is not None
    assert image_report.mask is None
    assert image_report.labels == []
    assert image_report.image.shape == [8, 9, 10]
    assert image_report.image.intensity["finite_voxel_count"] == 8 * 9 * 10

    mask_report = inspect_nifti(
        mask_path,
        role="mask",
        expected_labels={1},
    )
    assert mask_report.image is None
    assert mask_report.mask is not None
    assert len(mask_report.labels) == 1
    assert mask_report.labels[0].label == 1
    assert mask_report.labels[0].voxel_count > 0


def test_inspect_with_hash_records_sha256(tmp_path):
    image_path, _ = _write_pair(tmp_path)

    report = inspect_nifti(
        image_path,
        role="image",
        include_hash=True,
    )

    expected_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

    assert report.provenance["image"].sha256 == expected_hash
    assert report.provenance["image"].size_bytes == image_path.stat().st_size


def test_run_batch_with_relative_paths_and_expected_labels(tmp_path):
    image_path, mask_path = _write_pair(tmp_path)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,image_path,mask_path,expected_labels\n"
        f"case_001,data/{image_path.name},data/{mask_path.name},1\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "qa_report"

    batch = run_batch(
        manifest,
        output_dir,
        include_hash=True,
    )

    assert batch.total_cases == 1
    assert batch.error_cases == 0

    case = batch.cases[0]
    assert case.case_id == "case_001"
    assert len(case.labels) == 1
    assert case.labels[0].label == 1
    assert case.provenance["image"].sha256 is not None
    assert case.provenance["mask"].sha256 is not None

    assert (output_dir / "index.html").exists()
    assert (output_dir / "batch_report.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "label_statistics.csv").exists()
    assert (output_dir / "findings.jsonl").exists()
    assert (output_dir / "cases" / "case_001.json").exists()


def test_run_batch_converts_processing_failure_to_case_error(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,image_path,mask_path\n"
        "missing_case,missing_image.nii.gz,missing_mask.nii.gz\n",
        encoding="utf-8",
    )

    batch = run_batch(manifest, tmp_path / "qa_report")

    assert batch.total_cases == 1
    assert batch.error_cases == 1

    case = batch.cases[0]
    assert case.status == "error"
    assert len(case.findings) == 1
    assert case.findings[0].code == "case.processing_failed"
    assert case.findings[0].context["exception_type"] == "FileNotFoundError"


def test_run_batch_rejects_missing_required_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,image_path\n"
        "case_001,image.nii.gz\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required column"):
        run_batch(manifest, tmp_path / "qa_report")


def test_run_batch_rejects_empty_case_id(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,image_path,mask_path\n"
        ",image.nii.gz,mask.nii.gz\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty case_id"):
        run_batch(manifest, tmp_path / "qa_report")


def test_run_batch_rejects_duplicate_case_ids(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,image_path,mask_path\n"
        "case_001,image1.nii.gz,mask1.nii.gz\n"
        "case_001,image2.nii.gz,mask2.nii.gz\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        run_batch(manifest, tmp_path / "qa_report")