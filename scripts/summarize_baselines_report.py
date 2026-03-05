#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Make scripts runnable without requiring editable install when launched from outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.eval.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline outputs into paper-ready tables.")
    parser.add_argument("--baselines-root", required=True)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument(
        "--split-filter",
        default="all",
        help="all or comma-separated split names (train,val,test_id,test_ood_family,test_ood_severity,test_ood_hard_swap)",
    )
    return parser.parse_args()


def _parse_split_filter(raw: str) -> set[str] | None:
    if raw.strip().lower() == "all":
        return None
    splits = {s.strip() for s in raw.split(",") if s.strip()}
    return splits if splits else None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _aurc(curve: list[dict[str, float]]) -> float:
    if not curve:
        return 0.0
    if len(curve) == 1:
        return float(curve[0].get("risk", 0.0) * curve[0].get("coverage", 0.0))
    x = np.asarray([float(p.get("coverage", 0.0)) for p in curve], dtype=float)
    y = np.asarray([float(p.get("risk", 0.0)) for p in curve], dtype=float)
    return float(np.trapz(y, x))


def _risk_at_target(curve: list[dict[str, float]], target_coverage: float) -> float:
    if not curve:
        return 0.0
    for point in curve:
        if float(point.get("coverage", 0.0)) >= target_coverage:
            return float(point.get("risk", 0.0))
    return float(curve[-1].get("risk", 0.0))


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _render_markdown(rows: list[dict[str, str]]) -> str:
    header = [
        "baseline",
        "action_accuracy",
        "task_success",
        "coverage",
        "risk_at_80_task_success",
        "aurc_task_success",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in header) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baselines_root = Path(args.baselines_root)
    split_filter = _parse_split_filter(args.split_filter)
    target_coverage = float(args.target_coverage)

    if not baselines_root.exists():
        raise SystemExit(f"Baselines root does not exist: {baselines_root}")

    rows_for_csv: list[dict[str, str]] = []
    curves_out: dict[str, list[dict[str, float]]] = {}

    baseline_dirs = sorted(
        p for p in baselines_root.iterdir() if p.is_dir() and (p / "per_example_predictions.jsonl").exists()
    )
    if not baseline_dirs:
        raise SystemExit(f"No baseline prediction directories found under: {baselines_root}")

    for baseline_dir in baseline_dirs:
        baseline_name = baseline_dir.name
        preds_path = baseline_dir / "per_example_predictions.jsonl"
        records = _read_jsonl(preds_path)
        if split_filter is not None:
            records = [r for r in records if str(r.get("split", "")) in split_filter]
        if not records:
            continue

        metrics = summarize_metrics(records)
        curve = metrics.get("risk_coverage_task_success", [])
        if not isinstance(curve, list):
            curve = []
        curves_out[baseline_name] = curve

        row = {
            "baseline": baseline_name,
            "action_accuracy": _fmt(float(metrics.get("action_accuracy", 0.0))),
            "task_success": _fmt(float(metrics.get("task_success", 0.0))),
            "coverage": _fmt(float(metrics.get("coverage", 0.0))),
            "risk_at_80_task_success": _fmt(_risk_at_target(curve, target_coverage)),
            "aurc_task_success": _fmt(_aurc(curve)),
        }
        rows_for_csv.append(row)

    rows_for_csv.sort(key=lambda r: r["baseline"])

    report_dir = baselines_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "baseline_table.csv"
    md_path = report_dir / "baseline_table.md"
    curves_path = report_dir / "risk_coverage_task_success_curves.json"

    header = [
        "baseline",
        "action_accuracy",
        "task_success",
        "coverage",
        "risk_at_80_task_success",
        "aurc_task_success",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    md_path.write_text(_render_markdown(rows_for_csv), encoding="utf-8")
    curves_path.write_text(json.dumps(curves_out, indent=2), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {curves_path}")


if __name__ == "__main__":
    main()
