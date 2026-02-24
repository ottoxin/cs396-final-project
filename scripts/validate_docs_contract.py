#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC$")
DECISION_ID_RE = re.compile(r"^DEC-\d{4}$")
RUN_ID_RE = re.compile(r"RUN-\d{4}")


class ContractError(ValueError):
    """Raised when REPORT.md or LOG.md violates the docs contract."""


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ContractError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def _normalize_header_cell(cell: str) -> str:
    cleaned = cell.strip().strip("`").lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned


def _split_table_cells(row: str) -> list[str]:
    row = row.strip()
    if not row.startswith("|") or not row.endswith("|"):
        return []
    return [c.strip() for c in row.strip("|").split("|")]


def _find_section_line(lines: list[str], header: str) -> int:
    for idx, line in enumerate(lines):
        if line.strip() == header:
            return idx
    raise ContractError(f"Missing required section header: {header}")


def _find_table_after(lines: list[str], start_idx: int) -> tuple[list[str], list[list[str]]]:
    header_idx = -1
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip().startswith("|"):
            header_idx = i
            break
        if lines[i].startswith("## "):
            break
    if header_idx == -1:
        raise ContractError("Expected a markdown table after section header, but none was found.")

    if header_idx + 1 >= len(lines) or not lines[header_idx + 1].strip().startswith("|"):
        raise ContractError("Malformed markdown table: missing separator row.")

    header_cells = _split_table_cells(lines[header_idx])
    if not header_cells:
        raise ContractError("Malformed markdown table header row.")

    rows: list[list[str]] = []
    for i in range(header_idx + 2, len(lines)):
        raw = lines[i].strip()
        if not raw.startswith("|"):
            break
        cells = _split_table_cells(raw)
        if len(cells) != len(header_cells):
            raise ContractError(
                "Malformed markdown table row: column count mismatch "
                f"(expected {len(header_cells)}, got {len(cells)})."
            )
        rows.append(cells)
    return header_cells, rows


def _require_line_match(lines: list[str], prefix: str, regex: re.Pattern[str]) -> str:
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if not regex.match(value):
                raise ContractError(f"Invalid value for '{prefix}': {value}")
            return value
    raise ContractError(f"Missing required line with prefix: {prefix}")


def validate_report(report_path: Path) -> None:
    text = _read_text(report_path)
    lines = text.splitlines()

    required_sections = [
        "# REPORT.md - Phase A Results Ledger",
        "## Phase Status Snapshot",
        "## Run Ledger",
        "## Best-So-Far",
        "## Acceptance Gate Checklist",
        "## Known Regressions and Open Issues",
    ]
    for section in required_sections:
        _find_section_line(lines, section)

    _require_line_match(lines, "Last updated:", ISO_UTC_RE)

    snapshot_keys = [
        "- current_phase:",
        "- latest_dataset_manifest_id:",
        "- latest_baseline_summary_artifact:",
    ]
    for key in snapshot_keys:
        if not any(line.strip().startswith(key) for line in lines):
            raise ContractError(f"Missing required phase snapshot field: {key}")

    run_ledger_idx = _find_section_line(lines, "## Run Ledger")
    header_cells, rows = _find_table_after(lines, run_ledger_idx)
    expected_cols = [
        "run_id",
        "date_utc",
        "code_ref",
        "config_path",
        "dataset_manifest",
        "split_scope",
        "backbone_mode",
        "baseline_name",
        "acc",
        "action_acc",
        "macro_f1",
        "ece",
        "brier",
        "acceptance_pass",
        "artifact_paths",
    ]
    normalized_cols = [_normalize_header_cell(c) for c in header_cells]
    if normalized_cols != expected_cols:
        raise ContractError(
            "Run Ledger columns do not match required contract.\n"
            f"Expected: {expected_cols}\n"
            f"Found:    {normalized_cols}"
        )
    if not rows:
        raise ContractError("Run Ledger must include at least one row (placeholder row is allowed).")

    date_col_idx = expected_cols.index("date_utc")
    acceptance_col_idx = expected_cols.index("acceptance_pass")
    run_id_col_idx = expected_cols.index("run_id")
    for row in rows:
        date_value = row[date_col_idx]
        if not ISO_UTC_RE.match(date_value):
            raise ContractError(f"Invalid date_utc in Run Ledger row: {date_value}")

        acceptance = row[acceptance_col_idx].strip().lower()
        if acceptance not in {"yes", "no", "pending"}:
            raise ContractError(
                "Invalid acceptance_pass value in Run Ledger. Use one of: yes, no, pending."
            )

        run_id = row[run_id_col_idx].strip()
        if not run_id.startswith("RUN-"):
            raise ContractError(f"Invalid run_id format in Run Ledger row: {run_id}")

    best_idx = _find_section_line(lines, "## Best-So-Far")
    best_header, _ = _find_table_after(lines, best_idx)
    best_expected = [
        "split_scope",
        "baseline_name",
        "run_id",
        "acc",
        "action_acc",
        "macro_f1",
        "ece",
        "brier",
        "notes",
    ]
    if [_normalize_header_cell(c) for c in best_header] != best_expected:
        raise ContractError("Best-So-Far table columns do not match required contract.")

    gate_idx = _find_section_line(lines, "## Acceptance Gate Checklist")
    gate_header, _ = _find_table_after(lines, gate_idx)
    gate_expected = ["gate", "metric", "status", "notes"]
    if [_normalize_header_cell(c) for c in gate_header] != gate_expected:
        raise ContractError("Acceptance Gate Checklist columns do not match required contract.")


def _extract_log_entries(lines: list[str]) -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = []
    current_id = ""
    current_lines: list[str] = []
    for line in lines:
        heading_match = re.match(r"^###\s+(DEC-\d{4})(?:\s+.*)?$", line.strip())
        if heading_match:
            if current_id:
                entries.append((current_id, current_lines))
            current_id = heading_match.group(1)
            current_lines = []
            continue
        if current_id:
            current_lines.append(line)
    if current_id:
        entries.append((current_id, current_lines))
    return entries


def _parse_entry_fields(entry_lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in entry_lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def validate_log(log_path: Path) -> None:
    text = _read_text(log_path)
    lines = text.splitlines()

    required_sections = [
        "# LOG.md - Decision Record",
        "## Entry Template",
        "## Decision Log",
    ]
    for section in required_sections:
        _find_section_line(lines, section)

    _require_line_match(lines, "Last updated:", ISO_UTC_RE)

    entries = _extract_log_entries(lines)
    if not entries:
        raise ContractError("Decision Log must include at least one DEC entry.")

    required_fields = [
        "Date (UTC)",
        "Decision ID",
        "Area",
        "Context",
        "Options considered",
        "Decision",
        "Rationale",
        "Impact on PLAN/README/configs",
        "Follow-up actions",
    ]
    allowed_areas = {"data", "splits", "baselines", "eval", "infra", "docs"}
    parsed_times: list[datetime] = []

    for heading_id, body_lines in entries:
        fields = _parse_entry_fields(body_lines)
        missing = [k for k in required_fields if k not in fields]
        if missing:
            raise ContractError(
                f"Decision entry {heading_id} is missing required fields: {', '.join(missing)}"
            )

        date_text = fields["Date (UTC)"]
        if not ISO_UTC_RE.match(date_text):
            raise ContractError(f"Decision entry {heading_id} has invalid Date (UTC): {date_text}")
        parsed_times.append(datetime.strptime(date_text, "%Y-%m-%d %H:%M UTC"))

        decision_id = fields["Decision ID"]
        if not DECISION_ID_RE.match(decision_id):
            raise ContractError(f"Decision entry {heading_id} has invalid Decision ID format: {decision_id}")
        if decision_id != heading_id:
            raise ContractError(
                f"Decision entry heading/id mismatch: heading {heading_id}, field {decision_id}"
            )

        area = fields["Area"].strip().lower()
        if area not in allowed_areas:
            raise ContractError(
                f"Decision entry {heading_id} has invalid Area '{fields['Area']}'. "
                f"Allowed: {sorted(allowed_areas)}"
            )

        impact = fields["Impact on PLAN/README/configs"]
        if not any(token in impact for token in ["PLAN.md", "README.md", "config", "configs/"]):
            raise ContractError(
                f"Decision entry {heading_id} must reference PLAN/README/config impact explicitly."
            )

        joined_text = " ".join(
            [fields["Context"], fields["Decision"], fields["Rationale"]]
        ).lower()
        if "result" in joined_text:
            if "REPORT.md" not in impact and not RUN_ID_RE.search(impact):
                raise ContractError(
                    f"Decision entry {heading_id} appears result-motivated but lacks REPORT run reference."
                )

    for i in range(1, len(parsed_times)):
        if parsed_times[i] > parsed_times[i - 1]:
            raise ContractError(
                "Decision entries must be reverse chronological (newest first)."
            )


def _smoke_report(now_text: str) -> str:
    return f"""# REPORT.md - Phase A Results Ledger

Purpose: canonical experiment results ledger for Phase A.

Last updated: {now_text}

## Phase Status Snapshot
- current_phase: Phase A
- latest_dataset_manifest_id: manifest-placeholder
- latest_baseline_summary_artifact: artifacts/baselines/summary.json

## Run Ledger
| run_id | date_utc | code_ref | config_path | dataset_manifest | split_scope | backbone_mode | baseline_name | acc | action_acc | macro_f1 | ece | brier | acceptance_pass | artifact_paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUN-0001 | {now_text} | abc1234 | configs/default.yaml | manifest-placeholder | ID | mock | backbone_only | 0.50 | 0.50 | 0.10 | 0.20 | 0.30 | no | artifacts/baselines/run_0001 |

## Best-So-Far
| split_scope | baseline_name | run_id | acc | action_acc | macro_f1 | ece | brier | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID | backbone_only | RUN-0001 | 0.50 | 0.50 | 0.10 | 0.20 | 0.30 | smoke |

## Acceptance Gate Checklist
| gate | metric | status | notes |
| --- | --- | --- | --- |
| Accuracy | answer accuracy | pending | smoke |

## Known Regressions and Open Issues
- smoke entry
"""


def _smoke_log(now_text: str) -> str:
    return f"""# LOG.md - Decision Record

Purpose: chronological decision-and-reasoning record.

Last updated: {now_text}

## Entry Template
- Date (UTC): YYYY-MM-DD HH:MM UTC
- Decision ID: DEC-XXXX
- Area: data | splits | baselines | eval | infra | docs
- Context: Why this decision is needed now.
- Options considered: Enumerate concrete alternatives.
- Decision: Chosen option.
- Rationale: Why the chosen option is better.
- Impact on PLAN/README/configs: Reference exact files and sections changed.
- Follow-up actions: Concrete next steps.

## Decision Log

### DEC-0001 Smoke Validation Entry
- Date (UTC): {now_text}
- Decision ID: DEC-0001
- Area: docs
- Context: Smoke-check docs contract parsing.
- Options considered: Run without smoke mode vs run with smoke mode.
- Decision: Run with smoke mode and validate strict format.
- Rationale: Prevent malformed template edits.
- Impact on PLAN/README/configs: README.md documents validator command and PLAN.md contract remains unchanged.
- Follow-up actions: Run normal validation after smoke.
"""


def run_smoke() -> None:
    now_text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report_path = root / "REPORT.md"
        log_path = root / "LOG.md"
        report_path.write_text(_smoke_report(now_text), encoding="utf-8")
        log_path.write_text(_smoke_log(now_text), encoding="utf-8")
        validate_report(report_path)
        validate_log(log_path)
    print("Smoke validation passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate REPORT.md and LOG.md contract.")
    parser.add_argument(
        "--report",
        default="REPORT.md",
        help="Path to REPORT.md (optional unless --require-report is set)",
    )
    parser.add_argument(
        "--require-report",
        action="store_true",
        help="Fail if REPORT.md is missing.",
    )
    parser.add_argument(
        "--log",
        default="LOG.md",
        help="Path to LOG.md (optional unless --require-log is set)",
    )
    parser.add_argument(
        "--require-log",
        action="store_true",
        help="Fail if LOG.md is missing.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run synthetic smoke scenario with one run row and one decision entry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.smoke:
            run_smoke()
            return 0

        report_path = Path(args.report)
        log_path = Path(args.log)
        report_exists = report_path.exists()
        log_exists = log_path.exists()

        if report_exists:
            validate_report(report_path)
        elif args.require_report:
            raise ContractError(f"Missing required file: {report_path}")

        if log_exists:
            validate_log(log_path)
        elif args.require_log:
            raise ContractError(f"Missing required file: {log_path}")

        if report_exists and log_exists:
            print("Docs contract validation passed (REPORT.md and LOG.md).")
        elif report_exists and not log_exists:
            print("Docs contract validation passed (REPORT.md; LOG.md not present).")
        elif log_exists and not report_exists:
            print("Docs contract validation passed (LOG.md; REPORT.md not present).")
        else:
            print("Docs contract validation passed (REPORT.md and LOG.md not present).")
        return 0
    except ContractError as exc:
        print(f"Docs contract validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
