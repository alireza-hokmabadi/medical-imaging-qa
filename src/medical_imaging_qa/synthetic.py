from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required to create the synthetic demo dataset.") from exc
    return nib


def _sphere(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    grid = np.indices(shape, dtype=float)
    squared_distance = sum((grid[axis] - center[axis]) ** 2 for axis in range(3))
    return squared_distance <= radius**2


def create_demo_dataset(output_dir: str | Path) -> Path:
    """Create a fully synthetic three-case NIfTI dataset and return its manifest path."""
    nib = _nibabel()
    target = Path(output_dir)
    data_dir = target / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    shape = (64, 64, 32)
    affine = np.diag([1.25, 1.25, 2.5, 1.0])
    rng = np.random.default_rng(42)

    rows = []
    for case_id in ("clean", "fragmented", "geometry_mismatch"):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[_sphere(shape, (23, 31, 16), 9)] = 1
        mask[_sphere(shape, (42, 32, 16), 7)] = 2

        if case_id == "fragmented":
            mask[4:5, 4:5, 4:6] = 1
            mask[0:3, 28:31, 14:17] = 2

        image = rng.normal(0.0, 0.15, size=shape).astype(np.float32)
        image[mask == 1] += 1.5
        image[mask == 2] += 2.0

        mask_affine = affine.copy()
        if case_id == "geometry_mismatch":
            mask_affine[0, 3] = 3.0

        image_path = data_dir / f"{case_id}_image.nii.gz"
        mask_path = data_dir / f"{case_id}_mask.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), str(image_path))
        nib.save(nib.Nifti1Image(mask, mask_affine), str(mask_path))
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(image_path.relative_to(target)),
                "mask_path": str(mask_path.relative_to(target)),
                "expected_labels": "1;2",
            }
        )

    manifest = target / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["case_id", "image_path", "mask_path", "expected_labels"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest
