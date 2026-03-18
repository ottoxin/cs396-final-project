#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.data.io import load_examples, write_jsonl
from carm.data.schema import ConflictExample


EXPECTED_BY_CATEGORY: dict[str, dict[str, str]] = {
    "C1": {
        "vision_info_state": "informative",
        "text_info_state": "informative",
        "pairwise_relation": "consistent",
        "joint_action": "require_agreement",
        "joint_target": "gold",
    },
    "C2": {
        "vision_info_state": "informative",
        "text_info_state": "uninformative",
        "pairwise_relation": "asymmetric",
        "joint_action": "trust_vision",
        "joint_target": "gold",
    },
    "C3": {
        "vision_info_state": "uninformative",
        "text_info_state": "informative",
        "pairwise_relation": "asymmetric",
        "joint_action": "trust_text",
        "joint_target": "gold",
    },
    "C4": {
        "vision_info_state": "informative",
        "text_info_state": "informative",
        "pairwise_relation": "contradictory",
        "joint_action": "abstain",
        "joint_target": "<ABSTAIN>",
    },
    "C5": {
        "vision_info_state": "uninformative",
        "text_info_state": "uninformative",
        "pairwise_relation": "both_weak",
        "joint_action": "abstain",
        "joint_target": "<ABSTAIN>",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared dataset semantics against the revised C1-C5 label contract.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--examples_per_category", type=int, default=10)
    return parser.parse_args()


def _protocol_category(example: ConflictExample) -> str:
    if isinstance(example.metadata, dict):
        return str(example.metadata.get("protocol_category", "")).strip()
    return ""


def _joint_target_matches(example: ConflictExample, expected: str) -> bool:
    if expected == "gold":
        return example.joint_answer == example.gold_answer
    return str(example.joint_answer) == expected


def _field_available(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _example_sheet_row(example: ConflictExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "split": example.split.value,
        "protocol_category": _protocol_category(example),
        "question": example.question,
        "gold_answer": example.gold_answer,
        "vision_supported_target": example.vision_supported_target,
        "text_supported_target": example.text_supported_target,
        "vision_info_state": example.vision_info_state,
        "text_info_state": example.text_info_state,
        "pairwise_relation": example.pairwise_relation,
        "joint_action": example.oracle_action.value,
        "joint_target": example.joint_answer,
    }


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_category: dict[str, list[ConflictExample]] = defaultdict(list)
    for example in examples:
        by_category[_protocol_category(example)].append(example)

    completeness: dict[str, dict[str, Any]] = {}
    mapping_checks: dict[str, dict[str, Any]] = {}
    example_sheet_rows: list[dict[str, Any]] = []

    for category in sorted(EXPECTED_BY_CATEGORY):
        rows = by_category.get(category, [])
        expected = EXPECTED_BY_CATEGORY[category]

        completeness[category] = {
            "total_examples": len(rows),
            "vision_supported_target_available": sum(_field_available(row.vision_supported_target) for row in rows),
            "vision_supported_target_missing": sum(not _field_available(row.vision_supported_target) for row in rows),
            "text_supported_target_available": sum(_field_available(row.text_supported_target) for row in rows),
            "text_supported_target_missing": sum(not _field_available(row.text_supported_target) for row in rows),
            "vision_info_state_counts": dict(sorted(Counter(str(row.vision_info_state) for row in rows).items())),
            "text_info_state_counts": dict(sorted(Counter(str(row.text_info_state) for row in rows).items())),
            "pairwise_relation_counts": dict(sorted(Counter(str(row.pairwise_relation) for row in rows).items())),
            "joint_action_counts": dict(sorted(Counter(row.oracle_action.value for row in rows).items())),
            "joint_target_counts": dict(sorted(Counter(str(row.joint_answer) for row in rows).items())),
        }

        mapping_checks[category] = {
            "total_examples": len(rows),
            "vision_info_state_mismatches": sum((row.vision_info_state or "") != expected["vision_info_state"] for row in rows),
            "text_info_state_mismatches": sum((row.text_info_state or "") != expected["text_info_state"] for row in rows),
            "pairwise_relation_mismatches": sum((row.pairwise_relation or "") != expected["pairwise_relation"] for row in rows),
            "joint_action_mismatches": sum(row.oracle_action.value != expected["joint_action"] for row in rows),
            "joint_target_mismatches": sum(not _joint_target_matches(row, expected["joint_target"]) for row in rows),
        }

        for example in sorted(rows, key=lambda row: row.example_id)[: int(args.examples_per_category)]:
            example_sheet_rows.append(_example_sheet_row(example))

    completeness_path = output_dir / "field_completeness_by_category.json"
    mapping_path = output_dir / "category_mapping_checks.json"
    example_sheet_path = output_dir / "category_example_sheet.jsonl"
    report_path = output_dir / "data_sanity_report.md"

    completeness_path.write_text(json.dumps(completeness, indent=2), encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping_checks, indent=2), encoding="utf-8")
    write_jsonl(example_sheet_path, example_sheet_rows)

    lines = [
        "# Data Sanity Report",
        "",
        f"- input_jsonl: {args.input_jsonl}",
        f"- total_examples: {len(examples)}",
        "",
        "## Category Mapping Contract",
        "",
        "| category | vision_info_state | text_info_state | pairwise_relation | joint_action | joint_target |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category in ("C1", "C2", "C3", "C4", "C5"):
        expected = EXPECTED_BY_CATEGORY[category]
        lines.append(
            f"| {category} | {expected['vision_info_state']} | {expected['text_info_state']} | "
            f"{expected['pairwise_relation']} | {expected['joint_action']} | {expected['joint_target']} |"
        )

    lines.extend(["", "## Field Completeness by Category", ""])
    for category in ("C1", "C2", "C3", "C4", "C5"):
        stats = completeness[category]
        lines.append(f"### {category}")
        lines.append("")
        lines.append(f"- total_examples: {stats['total_examples']}")
        lines.append(
            f"- vision_supported_target: available={stats['vision_supported_target_available']}, missing={stats['vision_supported_target_missing']}"
        )
        lines.append(
            f"- text_supported_target: available={stats['text_supported_target_available']}, missing={stats['text_supported_target_missing']}"
        )
        lines.append(f"- vision_info_state_counts: {json.dumps(stats['vision_info_state_counts'], sort_keys=True)}")
        lines.append(f"- text_info_state_counts: {json.dumps(stats['text_info_state_counts'], sort_keys=True)}")
        lines.append(f"- pairwise_relation_counts: {json.dumps(stats['pairwise_relation_counts'], sort_keys=True)}")
        lines.append(f"- joint_action_counts: {json.dumps(stats['joint_action_counts'], sort_keys=True)}")
        lines.append(f"- joint_target_counts: {json.dumps(stats['joint_target_counts'], sort_keys=True)}")
        lines.append("")

    lines.extend(["## Mapping Verification", ""])
    for category in ("C1", "C2", "C3", "C4", "C5"):
        checks = mapping_checks[category]
        lines.append(f"### {category}")
        lines.append("")
        lines.append(f"- total_examples: {checks['total_examples']}")
        lines.append(f"- vision_info_state_mismatches: {checks['vision_info_state_mismatches']}")
        lines.append(f"- text_info_state_mismatches: {checks['text_info_state_mismatches']}")
        lines.append(f"- pairwise_relation_mismatches: {checks['pairwise_relation_mismatches']}")
        lines.append(f"- joint_action_mismatches: {checks['joint_action_mismatches']}")
        lines.append(f"- joint_target_mismatches: {checks['joint_target_mismatches']}")
        lines.append("")

    lines.extend(
        [
            "## Example Sheet",
            "",
            f"- saved_rows: {len(example_sheet_rows)}",
            f"- artifact: {example_sheet_path}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote report: {report_path}")
    print(f"wrote completeness: {completeness_path}")
    print(f"wrote mapping: {mapping_path}")
    print(f"wrote example sheet: {example_sheet_path}")


if __name__ == "__main__":
    main()
