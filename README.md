# Medical Imaging QA

[![CI](https://github.com/alireza-hokmabadi/medical-imaging-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/alireza-hokmabadi/medical-imaging-qa/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Medical Imaging QA (`miqa`)** is a Python package and command-line tool for reproducible quality assurance of NIfTI images and segmentation label maps.

It is designed for research pipelines where geometry errors, invalid labels, fragmented segmentations, or silent image/mask misalignment can propagate into downstream quantitative analysis.

The package produces both human-readable QA summaries and structured outputs suitable for automated pipelines.

## What it checks

### NIfTI geometry

- image and mask shape agreement
- voxel spacing agreement
- full affine-matrix agreement
- axis orientation consistency
- singular or invalid affines
- qform/sform availability

### Image integrity

- NaN and infinite values
- constant-intensity images
- robust intensity summary statistics

### Segmentation integrity

- 3D label-map validation
- non-finite and non-integer labels
- empty foreground masks
- missing and unexpected labels
- negative labels
- connected-component analysis
- small isolated components
- boundary contact

### Quantitative label statistics

For every foreground label, `miqa` reports:

- voxel count
- physical volume in mm³ and mL
- voxel-space centroid
- world-space centroid in mm
- voxel-space bounding box
- number of connected components
- largest-component size and fraction
- whether the label touches the image boundary

### Reproducibility and provenance

Optional SHA-256 hashing records the exact input files used for a QA run. Batch reports also preserve the QA-rule configuration used to produce the result.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/alireza-hokmabadi/medical-imaging-qa.git
cd medical-imaging-qa
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick start

Inspect one image:

```bash
miqa inspect image.nii.gz
```

Inspect a segmentation mask and declare expected labels:

```bash
miqa inspect mask.nii.gz --role mask -l 1 -l 2 -l 3
```

Validate an image/mask pair:

```bash
miqa validate image.nii.gz mask.nii.gz \
    --case-id subject_001 \
    -l 1 -l 2 -l 3 \
    --json-out subject_001_qa.json
```

Enable file hashing for stronger provenance:

```bash
miqa validate image.nii.gz mask.nii.gz --hash-files
```

## Batch QA

Create a CSV manifest:

```csv
case_id,image_path,mask_path,expected_labels
case_001,data/case_001_image.nii.gz,data/case_001_mask.nii.gz,1;2
case_002,data/case_002_image.nii.gz,data/case_002_mask.nii.gz,1;2
```

Run:

```bash
miqa batch manifest.csv --output-dir qa_report
```

The output directory contains:

```text
qa_report/
├── index.html
├── batch_report.json
├── summary.csv
├── label_statistics.csv
├── findings.jsonl
└── cases/
    ├── case_001.json
    └── case_002.json
```

`summary.csv` is convenient for cohort-level filtering, while `batch_report.json` and `findings.jsonl` are designed for programmatic use.

## Synthetic demo

The repository does not contain patient data. You can generate a fully synthetic NIfTI dataset with clean, fragmented, and deliberately misregistered examples:

```bash
miqa demo demo_output
```

This creates the synthetic images, masks, manifest, and a QA report in one command.

## Configurable QA rules

Thresholds can be stored in JSON:

```json
{
  "affine_atol": 0.0001,
  "spacing_atol": 0.00001,
  "integer_tolerance": 0.000001,
  "min_component_voxels": 10,
  "warn_multiple_components": true,
  "warn_boundary_touch": true,
  "warn_missing_qform_sform": true,
  "max_nonfinite_fraction": 0.0
}
```

Then use:

```bash
miqa batch manifest.csv -o qa_report --config qa_rules.json
```

## Pipeline-friendly exit codes

`miqa validate` and `miqa batch` can fail a CI or processing pipeline according to QA severity:

```bash
miqa validate image.nii.gz mask.nii.gz --fail-on error
miqa batch manifest.csv -o qa_report --fail-on warning
```

Use `--fail-on never` when reports should be generated without changing the process exit status.

## Python API

```python
from medical_imaging_qa import QARules, validate_pair

report = validate_pair(
    "image.nii.gz",
    "mask.nii.gz",
    case_id="subject_001",
    rules=QARules(min_component_voxels=20),
    expected_labels={1, 2, 3},
    include_hash=True,
)

print(report.status)
for label in report.labels:
    print(label.label, label.volume_ml)
```

## Stable finding codes

Each QA issue has a machine-readable code such as:

- `pair.affine_mismatch`
- `pair.spacing_mismatch`
- `mask.noninteger_labels`
- `mask.missing_expected_labels`
- `mask.multiple_components`
- `mask.small_components`
- `mask.touches_boundary`
- `image.nonfinite_values`

This makes it possible to build reliable downstream rules without parsing log messages.

## Development quality

The repository includes:

- unit tests for numerical QA logic
- NIfTI integration tests
- synthetic test data generated at runtime
- Ruff linting
- mypy type checking
- pytest coverage
- multi-version GitHub Actions CI

Run locally:

```bash
pytest
ruff check .
mypy src/medical_imaging_qa
```

## Intended use

This project is intended for **research and software quality assurance**. It does not determine whether a segmentation is clinically correct and does not replace expert visual review.

No patient or institutional data are included in this repository.

## License

MIT License. See [LICENSE](LICENSE).
