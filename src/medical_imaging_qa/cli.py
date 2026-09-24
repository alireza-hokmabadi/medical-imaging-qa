from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .api import inspect_nifti, run_batch, validate_pair
from .reporting import write_case_report
from .rules import QARules
from .synthetic import create_demo_dataset

app = typer.Typer(
    no_args_is_help=True,
    help="Reproducible QA for NIfTI images and segmentation label maps.",
)
console = Console()


class FailOn(str, Enum):
    NEVER = "never"
    WARNING = "warning"
    ERROR = "error"


def _rules(config: Path | None) -> QARules:
    return QARules.from_json(config) if config is not None else QARules()


def _expected(values: list[int] | None) -> set[int] | None:
    return set(values) if values else None


def _exit_code(status: str, fail_on: FailOn) -> int:
    if fail_on == FailOn.NEVER:
        return 0
    if fail_on == FailOn.WARNING and status in {"warning", "error"}:
        return 2
    if fail_on == FailOn.ERROR and status == "error":
        return 2
    return 0


def _render_case(report) -> None:
    console.print(
        f"[bold]Case:[/bold] {report.case_id}  "
        f"[bold]Status:[/bold] {report.status.upper()}"
    )
    if report.findings:
        table = Table("Severity", "Code", "Message")
        for finding in report.findings:
            table.add_row(finding.severity.value, finding.code, finding.message)
        console.print(table)
    if report.labels:
        table = Table("Label", "Voxels", "Volume (mL)", "Components", "Boundary")
        for item in report.labels:
            table.add_row(
                str(item.label),
                str(item.voxel_count),
                f"{item.volume_ml:.3f}",
                str(item.component_count),
                "yes" if item.touches_boundary else "no",
            )
        console.print(table)


@app.command("inspect")
def inspect_command(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    role: Annotated[str, typer.Option(help="Either 'image' or 'mask'.")] = "image",
    expected_label: Annotated[list[int] | None, typer.Option("--expected-label", "-l")] = None,
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    hash_file: Annotated[bool, typer.Option("--hash")] = False,
    json_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    if role not in {"image", "mask"}:
        raise typer.BadParameter("role must be 'image' or 'mask'.")
    report = inspect_nifti(
        path,
        role=role,
        rules=_rules(config),
        expected_labels=_expected(expected_label),
        include_hash=hash_file,
    )
    _render_case(report)
    if json_out is not None:
        write_case_report(report, json_out)


@app.command("validate")
def validate_command(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    mask: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    case_id: Annotated[str | None, typer.Option()] = None,
    expected_label: Annotated[list[int] | None, typer.Option("--expected-label", "-l")] = None,
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    hash_files: Annotated[bool, typer.Option()] = False,
    json_out: Annotated[Path | None, typer.Option()] = None,
    fail_on: Annotated[FailOn, typer.Option()] = FailOn.ERROR,
) -> None:
    report = validate_pair(
        image,
        mask,
        case_id=case_id,
        rules=_rules(config),
        expected_labels=_expected(expected_label),
        include_hash=hash_files,
    )
    _render_case(report)
    if json_out is not None:
        write_case_report(report, json_out)
    raise typer.Exit(_exit_code(report.status, fail_on))


@app.command("batch")
def batch_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    hash_files: Annotated[bool, typer.Option()] = False,
    fail_on: Annotated[FailOn, typer.Option()] = FailOn.ERROR,
) -> None:
    batch = run_batch(
        manifest,
        output_dir,
        rules=_rules(config),
        include_hash=hash_files,
    )
    console.print(
        f"Processed {batch.total_cases} case(s): "
        f"{batch.pass_cases} pass, {batch.warning_cases} warning, {batch.error_cases} error."
    )
    console.print(f"Report: {output_dir / 'index.html'}")
    if batch.error_cases:
        aggregate_status = "error"
    elif batch.warning_cases:
        aggregate_status = "warning"
    else:
        aggregate_status = "pass"
    raise typer.Exit(_exit_code(aggregate_status, fail_on))


@app.command("demo")
def demo_command(
    output_dir: Annotated[Path, typer.Argument()],
) -> None:
    manifest = create_demo_dataset(output_dir)
    report_dir = output_dir / "qa_report"
    batch = run_batch(manifest, report_dir)
    console.print(f"Synthetic demo created at {output_dir}")
    console.print(json.dumps(batch.to_dict()["summary"], indent=2))
    console.print(f"Open {report_dir / 'index.html'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
