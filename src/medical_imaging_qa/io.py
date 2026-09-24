from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import affine_determinant
from .models import ImageSummary


@dataclass
class LoadedNifti:
    path: Path
    data: np.ndarray
    shape: tuple[int, ...]
    dtype: str
    affine: np.ndarray
    zooms: tuple[float, ...]
    axcodes: tuple[str, ...]
    qform_code: int
    sform_code: int


def _nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency is installed in normal use
        raise RuntimeError(
            "nibabel is required for NIfTI I/O. Install medical-imaging-qa normally "
            "or run `pip install nibabel`."
        ) from exc
    return nib


def load_nifti(path: str | Path) -> LoadedNifti:
    nib = _nibabel()
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if not target.is_file():
        raise ValueError(f"Not a file: {target}")

    image = nib.load(str(target))
    data = np.asanyarray(image.dataobj)
    header = image.header
    qform_code = int(header["qform_code"])
    sform_code = int(header["sform_code"])

    return LoadedNifti(
        path=target,
        data=data,
        shape=tuple(int(value) for value in image.shape),
        dtype=str(data.dtype),
        affine=np.asarray(image.affine, dtype=float),
        zooms=tuple(float(value) for value in header.get_zooms()),
        axcodes=tuple(str(value) for value in nib.aff2axcodes(image.affine)),
        qform_code=qform_code,
        sform_code=sform_code,
    )


def to_summary(record: LoadedNifti, intensity: dict[str, float | int | None]) -> ImageSummary:
    return ImageSummary(
        path=str(record.path),
        shape=list(record.shape),
        dtype=record.dtype,
        zooms=list(record.zooms),
        axcodes=list(record.axcodes),
        affine=[[float(value) for value in row] for row in record.affine],
        affine_determinant=affine_determinant(record.affine),
        qform_code=record.qform_code,
        sform_code=record.sform_code,
        intensity=intensity,
    )
