# Design notes

Medical Imaging QA is deliberately split into three layers:

1. **Pure numerical QA** (`geometry.py`, `intensity.py`, `segmentation.py`) operates on NumPy arrays and affine matrices. This keeps the core logic testable without filesystem or NIfTI dependencies.
2. **NIfTI I/O** (`io.py`) is isolated behind a small adapter built on NiBabel.
3. **Workflow and reporting** (`api.py`, `reporting.py`, `cli.py`) handle manifests, provenance, machine-readable output, and command-line behaviour.

## QA philosophy

The package separates **findings** from **policy**. A finding has a stable code, severity, message, and structured context. Thresholds and selected checks live in `QARules`, making behaviour explicit and serialisable.

This is useful in research pipelines because downstream systems can consume stable finding codes rather than parse human-readable log strings.

## Geometry checks

Pair validation compares:

- array shape,
- voxel spacing,
- full 4x4 affine,
- axis orientation codes,
- affine validity,
- qform/sform availability.

The full affine check is intentionally stricter than comparing shape and spacing alone; two arrays can have identical dimensions while representing different physical locations.

## Segmentation checks

Label maps are checked for:

- dimensionality,
- non-finite values,
- non-integer values,
- foreground presence,
- expected/missing/unexpected labels,
- connected components,
- small isolated components,
- boundary contact.

Per-label output includes voxel count, physical volume, centroid in voxel and world coordinates, bounding box, component count, largest-component fraction, and boundary contact.

## Reproducibility

Optional SHA-256 hashing records exact input bytes. Reports also record file size and modification time. The JSON report stores the QA rule configuration used for the run.

## Scope

This package is a research QA utility, not a medical device. It does not determine whether a segmentation is clinically correct and it does not replace expert visual review.
