#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make scripts runnable without requiring editable install when launched from outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.data.schema import AnswerType
from carm.eval.canonicalization import CanonicalizationConfig, canonicalize_answer, semantic_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy v1 per-example predictions JSONL to v2 schema.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    return parser.parse_args()


def _guess_answer_type(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"boolean", "integer", "color", "unknown"}:
        return v
    return AnswerType.UNKNOWN.value


def main() -> None:
    args = parse_args()
    inp = Path(args.input_jsonl)
    out = Path(args.output_jsonl)

    cfg = CanonicalizationConfig()
    rows_out = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            answer_type = _guess_answer_type(str(row.get("answer_type", "unknown")))
            pred_raw = str(row.get("final_answer", ""))
            gold_raw = str(row.get("gold_answer", ""))

            pred = canonicalize_answer(pred_raw, answer_type, cfg=cfg)
            gold = canonicalize_answer(gold_raw, answer_type, cfg=cfg)

            exact = pred.normalized_text == gold.normalized_text and bool(pred.normalized_text)
            canonical = (
                pred.canonical_label is not None
                and gold.canonical_label is not None
                and pred.canonical_label == gold.canonical_label
            )
            semantic = semantic_match(pred.normalized_text, gold.normalized_text)

            migrated = {
                "schema_version": "2.0",
                "example": {
                    "example_id": row.get("example_id"),
                    "base_id": row.get("base_id"),
                    "variant_id": row.get("variant_id"),
                    "image_path": row.get("image_path"),
                    "text_input": row.get("text_input"),
                    "question": row.get("question"),
                    "split": row.get("split"),
                    "family": row.get("family"),
                    "operator": row.get("operator"),
                    "corrupt_modality": row.get("corrupt_modality"),
                    "severity": row.get("severity"),
                    "answer_type": answer_type,
                    "heldout_family_flag": row.get("heldout_family_flag"),
                    "heldout_severity_flag": row.get("heldout_severity_flag"),
                    "hard_swap_flag": row.get("hard_swap_flag"),
                    "metadata": row.get("metadata", {}),
                },
                "answer_output": {
                    "raw_text": pred_raw,
                    "normalized_text": pred.normalized_text,
                    "canonical_label": pred.canonical_label,
                    "canonical_status": pred.canonical_status,
                    "answer_confidence": row.get("confidence"),
                    "confidence_source": "legacy",
                    "metadata": {},
                },
                "policy_output": {
                    "pred_conflict_type": row.get("pred_conflict_type"),
                    "pred_action": row.get("pred_action"),
                    "abstained": row.get("abstained"),
                    "r_v": row.get("r_v"),
                    "r_t": row.get("r_t"),
                    "policy_confidence": row.get("confidence"),
                    "confidence_source": "legacy",
                    "audit": row.get("audit"),
                }
                if "pred_action" in row
                else None,
                "targets": {
                    "gold_answer": gold_raw,
                    "gold_normalized": gold.normalized_text,
                    "gold_canonical_label": gold.canonical_label,
                    "gold_canonical_status": gold.canonical_status,
                    "oracle_action": row.get("oracle_action"),
                    "target_r_v": row.get("target_r_v"),
                    "target_r_t": row.get("target_r_t"),
                },
                "derived": {
                    "exact_correct": exact,
                    "canonical_correct": canonical,
                    "semantic_correct": semantic,
                    "task_success": row.get("task_success"),
                },
            }
            fout.write(json.dumps(migrated, ensure_ascii=True) + "\n")
            rows_out += 1

    print(f"migrated {rows_out} rows -> {out}")


if __name__ == "__main__":
    main()
