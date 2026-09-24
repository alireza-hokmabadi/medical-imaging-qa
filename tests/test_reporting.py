import json

from medical_imaging_qa.models import BatchReport, CaseReport, Finding, Severity
from medical_imaging_qa.reporting import write_batch_report
from medical_imaging_qa.rules import QARules


def test_batch_report_writes_expected_outputs(tmp_path):
    case = CaseReport(
        case_id="case-1",
        status="warning",
        image=None,
        mask=None,
        findings=[Finding("test.warning", Severity.WARNING, "Example")],
    )
    batch = BatchReport([case])
    write_batch_report(batch, tmp_path, QARules())

    expected = {
        "batch_report.json",
        "summary.csv",
        "label_statistics.csv",
        "findings.jsonl",
        "index.html",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    payload = json.loads((tmp_path / "batch_report.json").read_text(encoding="utf-8"))
    assert payload["summary"]["warning_cases"] == 1
    assert (tmp_path / "cases" / "case-1.json").exists()
