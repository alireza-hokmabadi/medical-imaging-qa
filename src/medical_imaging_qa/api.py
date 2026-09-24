from __future__ import annotations

import csv
from pathlib import Path

from .geometry import compare_geometry, validate_affine
from .intensity import intensity_statistics, validate_intensity
from .io import load_nifti, to_summary
from .models import BatchReport, CaseReport, Finding, LabelStat, Severity, status_from_findings
from .provenance import file_provenance
from .reporting import write_batch_report
from .rules import QARules
from .segmentation import label_statistics, validate_labelmap


def _header_findings(record, role: str, rules: QARules) -> list[Finding]:
    findings = []
    for item in validate_affine(record.affine):
        suffix = item.code.split(".", maxsplit=1)[-1]
        findings.append(
            Finding(
                code=f"{role}.{suffix}",
                severity=item.severity,
                message=item.message,
                context=item.context,
            )
        )

    spatial_zooms = record.zooms[: min(3, len(record.zooms))]
    if any(value <= 0 for value in spatial_zooms):
        findings.append(
            Finding(
                code=f"{role}.invalid_spacing",
                severity=Severity.ERROR,
                message=f"{role.capitalize()} contains non-positive voxel spacing.",
                context={"zooms": list(spatial_zooms)},
            )
        )

    if rules.warn_missing_qform_sform and record.qform_code == 0 and record.sform_code == 0:
        findings.append(
            Finding(
                code=f"{role}.missing_qform_sform",
                severity=Severity.WARNING,
                message=f"{role.capitalize()} has neither qform nor sform code set.",
            )
        )
    return findings


def inspect_nifti(
    path: str | Path,
    role: str = "image",
    rules: QARules | None = None,
    expected_labels: set[int] | None = None,
    include_hash: bool = False,
) -> CaseReport:
    active_rules = rules or QARules()
    record = load_nifti(path)
    findings = _header_findings(record, role, active_rules)
    labels: list[LabelStat] = []

    if role == "mask":
        findings.extend(validate_labelmap(record.data, active_rules, expected_labels))
        has_blocking_mask_error = any(
            item.severity == Severity.ERROR and item.code.startswith("mask.") for item in findings
        )
        if not has_blocking_mask_error:
            labels, component_findings = label_statistics(record.data, record.affine, active_rules)
            findings.extend(component_findings)
        stats = {}
    else:
        findings.extend(validate_intensity(record.data, active_rules))
        stats = intensity_statistics(record.data)

    return CaseReport(
        case_id=record.path.stem,
        status=status_from_findings(findings),
        image=to_summary(record, stats) if role == "image" else None,
        mask=to_summary(record, stats) if role == "mask" else None,
        findings=findings,
        labels=labels,
        provenance={role: file_provenance(record.path, include_hash)},
    )


def validate_pair(
    image_path: str | Path,
    mask_path: str | Path,
    case_id: str | None = None,
    rules: QARules | None = None,
    expected_labels: set[int] | None = None,
    include_hash: bool = False,
) -> CaseReport:
    active_rules = rules or QARules()
    image = load_nifti(image_path)
    mask = load_nifti(mask_path)

    findings = _header_findings(image, "image", active_rules)
    findings.extend(_header_findings(mask, "mask", active_rules))
    findings.extend(validate_intensity(image.data, active_rules))
    findings.extend(validate_labelmap(mask.data, active_rules, expected_labels))
    findings.extend(
        compare_geometry(
            image.shape,
            image.affine,
            image.zooms,
            image.axcodes,
            mask.shape,
            mask.affine,
            mask.zooms,
            mask.axcodes,
            active_rules,
        )
    )

    labels: list[LabelStat] = []
    has_blocking_mask_error = any(
        item.severity == Severity.ERROR and item.code.startswith("mask.") for item in findings
    )
    if not has_blocking_mask_error:
        labels, component_findings = label_statistics(mask.data, mask.affine, active_rules)
        findings.extend(component_findings)

    image_stats = intensity_statistics(image.data)
    report = CaseReport(
        case_id=case_id or image.path.stem,
        status=status_from_findings(findings),
        image=to_summary(image, image_stats),
        mask=to_summary(mask, {}),
        findings=findings,
        labels=labels,
        provenance={
            "image": file_provenance(image.path, include_hash),
            "mask": file_provenance(mask.path, include_hash),
        },
    )
    return report


def _parse_expected_labels(raw: str) -> set[int] | None:
    value = raw.strip()
    if not value:
        return None
    return {int(item.strip()) for item in value.split(";") if item.strip()}


def run_batch(
    manifest_path: str | Path,
    output_dir: str | Path,
    rules: QARules | None = None,
    include_hash: bool = False,
) -> BatchReport:
    active_rules = rules or QARules()
    manifest = Path(manifest_path)
    base_dir = manifest.parent
    cases: list[CaseReport] = []
    seen_case_ids: set[str] = set()

    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "image_path", "mask_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Manifest is missing required column(s): {missing_text}")

        for row in reader:
            case_id = row["case_id"].strip()
            if not case_id:
                raise ValueError("Manifest contains an empty case_id.")
            if case_id in seen_case_ids:
                raise ValueError(f"Manifest contains duplicate case_id: {case_id}")
            seen_case_ids.add(case_id)

            image_path = Path(row["image_path"].strip())
            mask_path = Path(row["mask_path"].strip())
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            if not mask_path.is_absolute():
                mask_path = base_dir / mask_path
            expected = _parse_expected_labels(row.get("expected_labels", ""))

            try:
                report = validate_pair(
                    image_path=image_path,
                    mask_path=mask_path,
                    case_id=case_id,
                    rules=active_rules,
                    expected_labels=expected,
                    include_hash=include_hash,
                )
            except Exception as exc:
                finding = Finding(
                    code="case.processing_failed",
                    severity=Severity.ERROR,
                    message="Case could not be processed.",
                    context={"exception_type": type(exc).__name__, "detail": str(exc)},
                )
                report = CaseReport(
                    case_id=case_id,
                    status="error",
                    image=None,
                    mask=None,
                    findings=[finding],
                )
            cases.append(report)

    batch = BatchReport(cases=cases)
    write_batch_report(batch, output_dir, active_rules)
    return batch
