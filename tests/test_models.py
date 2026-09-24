from medical_imaging_qa.models import Finding, Severity, status_from_findings


def test_status_precedence():
    assert status_from_findings([]) == "pass"
    assert status_from_findings([Finding("w", Severity.WARNING, "warning")]) == "warning"
    assert status_from_findings(
        [
            Finding("w", Severity.WARNING, "warning"),
            Finding("e", Severity.ERROR, "error"),
        ]
    ) == "error"
