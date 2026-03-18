#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_JSONL = PROJECT_ROOT / "data" / "cache" / "hf_5way" / "prepared" / "carm_vqa_5way_10pct_protocol_family_seed7.jsonl"
DEFAULT_ROOTS = [
    PROJECT_ROOT / "outputs" / "experimental" / "RUN-EXP-0007_10pct_qwen_protocol",
    PROJECT_ROOT / "outputs" / "carm" / "RUN-CTRL-0001_10pct_protocol",
]
TARGET_FILE_NAMES = {
    "per_example_predictions.jsonl",
    "per_example_predictions.csv",
    "val_predictions_best.jsonl",
    "failure_diagnostics.csv",
}
FIELD_ORDER = [
    "protocol_category",
    "oracle_action",
    "gold_action_legacy",
    "derived_action_target",
    "derived_action_target_available",
    "derived_joint_info_state",
    "derived_pairwise_relation",
    "derived_text_info_state",
    "derived_vision_info_state",
    "vision_supported_target",
    "text_supported_target",
    "vision_info_state",
    "text_info_state",
    "pairwise_relation",
    "joint_answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill saved run artifacts with HF-aligned protocol semantics from the corrected prepared subset.")
    parser.add_argument("--prepared-jsonl", type=Path, default=DEFAULT_PREPARED_JSONL)
    parser.add_argument("--roots", type=Path, nargs="+", default=DEFAULT_ROOTS)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _joint_info_state(vision_info: str | None, text_info: str | None) -> str | None:
    if not vision_info or not text_info:
        return None
    return f"vision_{vision_info}__text_{text_info}"


def _prepared_lookup(prepared_jsonl: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(prepared_jsonl):
        metadata = row.get("metadata") or {}
        protocol_category = metadata.get("protocol_category") or row.get("protocol_category")
        oracle_action = row.get("oracle_action")
        vision_info_state = row.get("vision_info_state")
        text_info_state = row.get("text_info_state")
        lookup[row["example_id"]] = {
            "protocol_category": protocol_category,
            "oracle_action": oracle_action,
            "gold_action_legacy": oracle_action,
            "derived_action_target": oracle_action,
            "derived_action_target_available": oracle_action is not None,
            "derived_joint_info_state": _joint_info_state(vision_info_state, text_info_state),
            "derived_pairwise_relation": row.get("pairwise_relation"),
            "derived_text_info_state": text_info_state,
            "derived_vision_info_state": vision_info_state,
            "vision_supported_target": row.get("vision_supported_target"),
            "text_supported_target": row.get("text_supported_target"),
            "vision_info_state": vision_info_state,
            "text_info_state": text_info_state,
            "pairwise_relation": row.get("pairwise_relation"),
            "joint_answer": row.get("joint_answer"),
        }
    return lookup


def _update_row(row: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    for key, value in canonical.items():
        updated[key] = value
    return updated


def _rewrite_jsonl(path: Path, lookup: dict[str, dict[str, Any]]) -> tuple[int, int]:
    rows = _load_jsonl(path)
    changed = 0
    matched = 0
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        example_id = row.get("example_id")
        canonical = lookup.get(example_id)
        if canonical is None:
            updated_rows.append(row)
            continue
        matched += 1
        new_row = _update_row(row, canonical)
        if new_row != row:
            changed += 1
        updated_rows.append(new_row)
    if changed:
        _write_jsonl(path, updated_rows)
    return matched, changed


def _rewrite_csv(path: Path, lookup: dict[str, dict[str, Any]]) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "example_id" not in fieldnames:
        return 0, 0

    for field in FIELD_ORDER:
        if field not in fieldnames:
            fieldnames.append(field)

    changed = 0
    matched = 0
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        example_id = row.get("example_id")
        canonical = lookup.get(example_id)
        if canonical is None:
            updated_rows.append(row)
            continue
        matched += 1
        new_row = dict(row)
        before = dict(row)
        for key, value in canonical.items():
            if key in fieldnames:
                new_row[key] = "" if value is None else str(value)
        if new_row != before:
            changed += 1
        updated_rows.append(new_row)

    if changed:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
    return matched, changed


def main() -> int:
    args = parse_args()
    lookup = _prepared_lookup(args.prepared_jsonl)
    summary: list[str] = []
    for root in args.roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name not in TARGET_FILE_NAMES:
                continue
            if path.suffix == ".jsonl":
                matched, changed = _rewrite_jsonl(path, lookup)
            elif path.suffix == ".csv":
                matched, changed = _rewrite_csv(path, lookup)
            else:
                continue
            summary.append(f"{path}: matched={matched} changed={changed}")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
