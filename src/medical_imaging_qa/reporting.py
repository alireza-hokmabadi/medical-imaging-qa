from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict
from pathlib import Path

from .models import BatchReport, CaseReport
from .rules import QARules
from .version import __version__


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_case_report(report: CaseReport, path: str | Path) -> None:
    _write_json(
        Path(path),
        {
            "software": {"name": "medical-imaging-qa", "version": __version__},
            **report.to_dict(),
        },
    )


def _write_summary_csv(batch: BatchReport, path: Path) -> None:
    fields = ["case_id", "status", "error_count", "warning_count", "label_count"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in batch.cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "status": case.status,
                    "error_count": sum(item.severity.value == "error" for item in case.findings),
                    "warning_count": sum(
                        item.severity.value == "warning" for item in case.findings
                    ),
                    "label_count": len(case.labels),
                }
            )


def _write_label_csv(batch: BatchReport, path: Path) -> None:
    fields = [
        "case_id",
        "label",
        "voxel_count",
        "volume_mm3",
        "volume_ml",
        "component_count",
        "largest_component_voxels",
        "largest_component_fraction",
        "touches_boundary",
        "centroid_voxel",
        "centroid_world_mm",
        "bbox_min_voxel",
        "bbox_max_voxel",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in batch.cases:
            for item in case.labels:
                row = asdict(item)
                row["case_id"] = case.case_id
                vector_fields = (
                    "centroid_voxel",
                    "centroid_world_mm",
                    "bbox_min_voxel",
                    "bbox_max_voxel",
                )
                for key in vector_fields:
                    row[key] = json.dumps(row[key], separators=(",", ":"))
                writer.writerow(row)


def _write_findings_jsonl(batch: BatchReport, path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for case in batch.cases:
            for finding in case.findings:
                payload = {"case_id": case.case_id, **finding.to_dict()}
                stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _html_report(batch: BatchReport) -> str:
    rows = []
    for case in batch.cases:
        error_count = sum(item.severity.value == "error" for item in case.findings)
        warning_count = sum(item.severity.value == "warning" for item in case.findings)
        rows.append(
            "<tr>"
            f"<td>{html.escape(case.case_id)}</td>"
            f"<td class='{html.escape(case.status)}'>{html.escape(case.status.upper())}</td>"
            f"<td>{error_count}</td><td>{warning_count}</td><td>{len(case.labels)}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Medical Imaging QA Report</title>
<style>
body {{
  font-family: system-ui, sans-serif; max-width: 1100px;
  margin: 2rem auto; padding: 0 1rem;
}}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: .55rem; text-align: left; }}
.pass {{ font-weight: 700; }} .warning {{ font-weight: 700; }} .error {{ font-weight: 700; }}
.summary {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0 2rem; }}
.card {{ border: 1px solid #ddd; border-radius: .5rem; padding: .8rem 1rem; min-width: 130px; }}
code {{ background: #f2f2f2; padding: .1rem .25rem; border-radius: .2rem; }}
</style>
</head>
<body>
<h1>Medical Imaging QA Report</h1>
<div class="summary">
<div class="card"><strong>Total</strong><br>{batch.total_cases}</div>
<div class="card"><strong>Pass</strong><br>{batch.pass_cases}</div>
<div class="card"><strong>Warnings</strong><br>{batch.warning_cases}</div>
<div class="card"><strong>Errors</strong><br>{batch.error_cases}</div>
</div>
<p>Machine-readable details are available in <code>batch_report.json</code>,
<code>findings.jsonl</code>, and <code>label_statistics.csv</code>.</p>
<table>
<thead><tr><th>Case</th><th>Status</th><th>Errors</th><th>Warnings</th><th>Labels</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def write_batch_report(batch: BatchReport, output_dir: str | Path, rules: QARules) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cases_dir = target / "cases"
    cases_dir.mkdir(exist_ok=True)

    _write_json(
        target / "batch_report.json",
        {
            "software": {"name": "medical-imaging-qa", "version": __version__},
            **batch.to_dict(),
            "rules": rules.to_dict(),
        },
    )
    _write_summary_csv(batch, target / "summary.csv")
    _write_label_csv(batch, target / "label_statistics.csv")
    _write_findings_jsonl(batch, target / "findings.jsonl")
    (target / "index.html").write_text(_html_report(batch), encoding="utf-8")

    for case in batch.cases:
        safe_name = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in case.case_id
        )
        _write_json(
            cases_dir / f"{safe_name}.json",
            {
                "software": {"name": "medical-imaging-qa", "version": __version__},
                **case.to_dict(),
            },
        )
