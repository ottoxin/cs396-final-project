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
    parser = argparse.ArgumentParser(description="Summarize flattened baseline outputs into report tables.")
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


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def _render_markdown(rows: list[dict[str, str]], header: list[str]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in header) + " |")
    return "\n".join(lines) + "\n"


def _row_split(row: dict) -> str:
    return str(row.get("split", ""))


def summarize_baselines_root(
    baselines_root: str | Path,
    *,
    target_coverage: float = 0.8,
    split_filter: str = "all",
) -> dict[str, Path]:
    baselines_root = Path(baselines_root)
    split_filter_set = _parse_split_filter(split_filter)
    if not baselines_root.exists():
        raise SystemExit(f"Baselines root does not exist: {baselines_root}")

    main_rows: list[dict[str, str]] = []
    category_rows: list[dict[str, str]] = []
    c2_rows: list[dict[str, str]] = []
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
        if split_filter_set is not None:
            records = [r for r in records if _row_split(r) in split_filter_set]
        if not records:
            continue

        metrics = summarize_metrics(records)
        curve = metrics.get("risk_coverage_task_success", [])
        if not isinstance(curve, list):
            curve = []
        curves_out[baseline_name] = curve

        main_rows.append(
            {
                "baseline": baseline_name,
                "task_success": _fmt(float(metrics.get("task_success", 0.0))),
                "accuracy": _fmt(float(metrics.get("accuracy", 0.0))),
                "coverage": _fmt(float(metrics.get("coverage", 0.0))),
                "acc_on_answered": _fmt(float(metrics.get("accuracy_on_answered", 0.0))),
                "final_unknown": _fmt(float(metrics.get("final_unknown_rate", 0.0))),
                "fallback": _fmt(float(metrics.get("fallback_rate", 0.0))),
                "oov_gen": _fmt(float(metrics.get("out_of_vocab_generation_rate", 0.0))),
                "parsed_argmax": _fmt(float(metrics.get("parsed_argmax_agreement_rate", 0.0))),
                "risk@80_ts": _fmt(_risk_at_target(curve, target_coverage)),
                "aurc_ts": _fmt(_aurc(curve)),
            }
        )

        per_category = metrics.get("task_success_per_category", {})
        if not isinstance(per_category, dict):
            per_category = {}
        category_rows.append(
            {
                "baseline": baseline_name,
                "C1": _fmt(float(per_category.get("C1", 0.0))),
                "C2": _fmt(float(per_category.get("C2", 0.0))),
                "C3": _fmt(float(per_category.get("C3", 0.0))),
                "C4": _fmt(float(per_category.get("C4", 0.0))),
                "C5": _fmt(float(per_category.get("C5", 0.0))),
            }
        )
        c2_rows.append(
            {
                "baseline": baseline_name,
                "vision_only_acc": _fmt(metrics.get("c2_vision_only_accuracy")),
                "text_only_acc": _fmt(metrics.get("c2_text_only_accuracy")),
                "multimodal_abst_rate": _fmt(metrics.get("c2_multimodal_abstention_rate")),
            }
        )

    main_rows.sort(key=lambda r: r["baseline"])
    category_rows.sort(key=lambda r: r["baseline"])
    c2_rows.sort(key=lambda r: r["baseline"])

    report_dir = baselines_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    main_csv = report_dir / "main_table.csv"
    main_md = report_dir / "main_table.md"
    category_csv = report_dir / "per_category_task_success.csv"
    category_md = report_dir / "per_category_task_success.md"
    c2_csv = report_dir / "c2_diagnostics.csv"
    c2_md = report_dir / "c2_diagnostics.md"
    curves_json = report_dir / "risk_coverage_task_success_curves.json"

    main_header = [
        "baseline",
        "task_success",
        "accuracy",
        "coverage",
        "acc_on_answered",
        "final_unknown",
        "fallback",
        "oov_gen",
        "parsed_argmax",
        "risk@80_ts",
        "aurc_ts",
    ]
    with main_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=main_header)
        writer.writeheader()
        writer.writerows(main_rows)
    main_md.write_text(_render_markdown(main_rows, main_header), encoding="utf-8")

    category_header = ["baseline", "C1", "C2", "C3", "C4", "C5"]
    with category_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=category_header)
        writer.writeheader()
        writer.writerows(category_rows)
    category_md.write_text(_render_markdown(category_rows, category_header), encoding="utf-8")

    c2_header = ["baseline", "vision_only_acc", "text_only_acc", "multimodal_abst_rate"]
    with c2_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=c2_header)
        writer.writeheader()
        writer.writerows(c2_rows)
    c2_md.write_text(_render_markdown(c2_rows, c2_header), encoding="utf-8")
    curves_json.write_text(json.dumps(curves_out, indent=2), encoding="utf-8")

    return {
        "main_csv": main_csv,
        "main_md": main_md,
        "category_csv": category_csv,
        "category_md": category_md,
        "c2_csv": c2_csv,
        "c2_md": c2_md,
        "curves_json": curves_json,
    }


def main() -> None:
    args = parse_args()
    outputs = summarize_baselines_root(
        args.baselines_root,
        target_coverage=float(args.target_coverage),
        split_filter=args.split_filter,
    )
    for path in outputs.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
