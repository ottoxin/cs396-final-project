#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from carm.data.hf5way import CATEGORY_TO_ACTION, STATE_TO_CATEGORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit derivability from the raw nbso/carm-vqa-5way dataset before any training."
    )
    parser.add_argument("--hf-repo-id", default="nbso/carm-vqa-5way")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument(
        "--prepared-jsonl",
        default=str(PROJECT_ROOT / "data/cache/hf_5way/prepared/carm_vqa_5way.jsonl"),
    )
    parser.add_argument(
        "--prepared-manifest",
        default=str(PROJECT_ROOT / "data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json"),
    )
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_raw_schema(repo_id: str, revision: str) -> dict[str, Any]:
    dataset = load_dataset(repo_id, revision=revision)
    split_rows = {split: int(len(rows)) for split, rows in dataset.items()}
    split_fields = {split: list(rows.features.keys()) for split, rows in dataset.items()}
    common_fields = sorted(set.intersection(*(set(fields) for fields in split_fields.values())))
    split_examples: dict[str, dict[str, Any]] = {}
    for split, rows in dataset.items():
        row = rows[0]
        split_examples[split] = {
            key: row[key]
            for key in common_fields
            if key != "image_path"
        }
    return {
        "repo_id": repo_id,
        "revision": revision,
        "split_rows": split_rows,
        "split_fields": split_fields,
        "common_fields": common_fields,
        "split_examples": split_examples,
        "total_rows": int(sum(split_rows.values())),
    }


def _load_prepared_contradiction_stats(path: Path) -> dict[str, Any]:
    contradiction_total = 0
    contradiction_missing = 0
    contradiction_same_as_gold = 0
    contradiction_diff_from_gold = 0
    by_family: dict[str, Counter[str]] = {}
    source_counter: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = row.get("metadata", {}) or {}
            if str(metadata.get("protocol_category", "")).strip().upper() != "C4":
                continue
            contradiction_total += 1
            family = str(row.get("family", "unknown"))
            family_counter = by_family.setdefault(family, Counter())
            family_counter["total"] += 1

            source = str(metadata.get("text_supported_target_source", "none"))
            source_counter[source] += 1
            family_counter[f"source:{source}"] += 1

            gold_answer = str(row.get("gold_answer", "")).strip()
            text_target = row.get("text_supported_target")
            text_target = str(text_target).strip() if text_target is not None else ""
            if not text_target:
                contradiction_missing += 1
                family_counter["missing"] += 1
                continue
            if text_target == gold_answer:
                contradiction_same_as_gold += 1
                family_counter["same_as_gold"] += 1
            else:
                contradiction_diff_from_gold += 1
                family_counter["diff_from_gold"] += 1

    return {
        "contradiction_total": contradiction_total,
        "contradiction_missing": contradiction_missing,
        "contradiction_same_as_gold": contradiction_same_as_gold,
        "contradiction_diff_from_gold": contradiction_diff_from_gold,
        "text_target_source_counts": dict(sorted(source_counter.items())),
        "by_family": {family: dict(sorted(counter.items())) for family, counter in sorted(by_family.items())},
    }


def _rule_lines() -> list[str]:
    lines = [
        "`(image_state, caption_state) -> protocol category`",
    ]
    for states, category in sorted(STATE_TO_CATEGORY.items()):
        lines.append(f"- `{states} -> {category}`")
    lines.append("")
    lines.append("`protocol category -> legacy action`")
    for category, action in sorted(CATEGORY_TO_ACTION.items()):
        lines.append(f"- `{category} -> {action}`")
    lines.append("")
    lines.append("`Conservative evidential rules used in this audit`")
    lines.append("- `vision informative` if `image_state == clean`; `vision uninformative` if `image_state == irrelevant`.")
    lines.append("- `text uninformative` if `caption_state == irrelevant`.")
    lines.append("- `text supportive` if `caption_state == clean`, because the raw row serves the clean caption and the gold answer is defined against that row.")
    lines.append(
        "- `caption_state == different` is treated as only a candidate contradiction bucket. The raw row has no explicit `text_supported_target`, so answer-grounded contradiction is not fully derivable from raw fields alone."
    )
    lines.append("- `vision target` can only be proxied by `gold_answer` on rows with `image_state == clean`; otherwise it must be masked.")
    lines.append(
        "- `text target` can use `gold_answer` on rows with `caption_state == clean`; `caption_state == different` needs an extra caption-to-answer extraction rule and is therefore only heuristic unless a source field exists."
    )
    return lines


def _build_target_audit(manifest: dict[str, Any], contradiction_stats: dict[str, Any]) -> list[dict[str, Any]]:
    counts = manifest["protocol_category_counts"]
    total_rows = int(manifest["total_rows_written"])
    c1 = int(counts.get("C1", 0))
    c2 = int(counts.get("C2", 0))
    c3 = int(counts.get("C3", 0))
    c4 = int(counts.get("C4", 0))
    c5 = int(counts.get("C5", 0))
    clean_image_rows = c1 + c2 + c4
    irrelevant_image_rows = c3 + c5
    clean_text_rows = c1 + c3
    irrelevant_text_rows = c2 + c5
    non_different_rows = c1 + c2 + c3 + c5

    return [
        {
            "target": "vision informative / uninformative",
            "fields_used": ["image_state"],
            "rule": "`clean -> informative`, `irrelevant -> uninformative`",
            "status": "fully derivable",
            "coverage": f"{total_rows}/{total_rows}",
            "masking": "none",
        },
        {
            "target": "text informative / uninformative",
            "fields_used": ["caption_state", "gold_answer"],
            "rule": (
                "`irrelevant -> uninformative`; `clean -> informative/supportive`; "
                "`different` is not enough to prove answer-grounded informativeness or contradiction"
            ),
            "status": "partially derivable",
            "coverage": f"{non_different_rows}/{total_rows} fully resolvable; {c4} candidate `different` rows ambiguous",
            "masking": f"mask `caption_state == different` ({c4} rows) if you need answer-grounded informativeness",
        },
        {
            "target": "pairwise consistent",
            "fields_used": ["image_state", "caption_state"],
            "rule": "`(clean, clean) -> consistent`",
            "status": "fully derivable",
            "coverage": str(c1),
            "masking": "none",
        },
        {
            "target": "pairwise contradictory",
            "fields_used": ["image_state", "caption_state", "question", "perturbed_caption"],
            "rule": "`(clean, different)` is only a candidate contradiction bucket from raw fields",
            "status": "partially derivable",
            "coverage": (
                f"{c4} raw candidate rows; prepared heuristic later filled {contradiction_stats['contradiction_diff_from_gold']} "
                f"text targets that differ from gold, {contradiction_stats['contradiction_same_as_gold']} that match gold, "
                f"and missed {contradiction_stats['contradiction_missing']}"
            ),
            "masking": "mask all raw `different` rows if contradiction must be answer-grounded rather than protocol-defined",
        },
        {
            "target": "pairwise asymmetric",
            "fields_used": ["image_state", "caption_state"],
            "rule": "`(clean, irrelevant)` or `(irrelevant, clean) -> asymmetric`",
            "status": "fully derivable",
            "coverage": str(c2 + c3),
            "masking": "none",
        },
        {
            "target": "pairwise both weak",
            "fields_used": ["image_state", "caption_state"],
            "rule": "`(irrelevant, irrelevant) -> both weak`",
            "status": "fully derivable",
            "coverage": str(c5),
            "masking": "none",
        },
        {
            "target": "vision target",
            "fields_used": ["image_state", "gold_answer"],
            "rule": "`gold_answer` is the only defensible proxy when `image_state == clean`",
            "status": "partially derivable",
            "coverage": f"{clean_image_rows}/{total_rows} conditionally available",
            "masking": f"mask `image_state == irrelevant` ({irrelevant_image_rows} rows)",
        },
        {
            "target": "text target",
            "fields_used": ["caption_state", "gold_answer", "question", "perturbed_caption"],
            "rule": (
                "`gold_answer` is available when `caption_state == clean`; "
                "`caption_state == different` needs an extra caption-to-answer rule absent from the raw row"
            ),
            "status": "partially derivable",
            "coverage": (
                f"{clean_text_rows}/{total_rows} clean-caption rows directly usable; "
                f"{contradiction_stats['contradiction_total']} `different` rows need heuristics"
            ),
            "masking": (
                f"mask `caption_state == irrelevant` ({irrelevant_text_rows} rows); "
                f"also mask raw `different` rows unless you explicitly accept heuristic targets"
            ),
        },
        {
            "target": "joint action (legacy protocol)",
            "fields_used": ["oracle_action", "category", "image_state", "caption_state"],
            "rule": "use raw `oracle_action`, or reconstruct through state/category mapping",
            "status": "fully derivable",
            "coverage": f"{total_rows}/{total_rows}",
            "masking": "none",
        },
        {
            "target": "joint action (conservative evidential)",
            "fields_used": ["image_state", "caption_state", "gold_answer"],
            "rule": (
                "`(clean, clean) -> require_agreement`; `(clean, irrelevant) -> trust_vision`; "
                "`(irrelevant, clean) -> trust_text`; `(irrelevant, irrelevant) -> abstain`"
            ),
            "status": "partially derivable",
            "coverage": f"{non_different_rows}/{total_rows} fully defensible; {c4} `different` rows should be masked",
            "masking": f"mask all `caption_state == different` rows ({c4})",
        },
    ]


def _build_markdown(
    raw_schema: dict[str, Any],
    manifest: dict[str, Any],
    contradiction_stats: dict[str, Any],
    target_audit: list[dict[str, Any]],
) -> str:
    common_fields = ", ".join(f"`{field}`" for field in raw_schema["common_fields"])
    split_rows = ", ".join(f"`{split}={rows}`" for split, rows in raw_schema["split_rows"].items())
    contradiction_total = contradiction_stats["contradiction_total"]
    same_cardinality = raw_schema["total_rows"] == int(manifest["total_rows_written"])

    lines = [
        "# Raw HF Derivability Audit",
        "",
        "## Scope",
        "",
        f"- Raw dataset: `{raw_schema['repo_id']}` @ `{manifest.get('hf_sha', raw_schema['revision'])}`",
        f"- Raw splits: {split_rows}",
        f"- Raw common fields used: {common_fields}",
        f"- Prepared mirror rows: `{manifest['total_rows_written']}`",
        f"- Raw/prepared row cardinality match: `{same_cardinality}`",
        f"- Prepared drop counts: `{json.dumps(manifest.get('drop_counts', {}), sort_keys=True)}`",
        "",
        "## Exact Derivation Rules",
        "",
        *_rule_lines(),
        "",
        "## Requested Target Audit",
        "",
        "| requested target | source fields used | exact conservative rule | status | coverage | mask recommendation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in target_audit:
        fields = ", ".join(f"`{field}`" for field in row["fields_used"])
        lines.append(
            f"| {row['target']} | {fields} | {row['rule']} | {row['status']} | {row['coverage']} | {row['masking']} |"
        )

    lines.extend(
        [
            "",
            "## What Raw Data Does Not Provide",
            "",
            "- No raw `protocol_category` field. It must be reconstructed from `(image_state, caption_state)`.",
            "- No raw `vision_supported_target` field.",
            "- No raw `text_supported_target` field.",
            "- No raw modality-specific answer/confidence fields for vision-only or text-only behavior.",
            "",
            "## Prepared-Stage Implication for C4 / `caption_state == different`",
            "",
            f"- Candidate raw `different` rows: `{contradiction_total}`.",
            f"- Preparation added `vision_supported_target` on all contradiction rows by copying `gold_answer`: `{manifest['contradiction_target_counts']['vision_supported_target']}`.",
            (
                f"- Preparation added `text_supported_target` on `{manifest['contradiction_target_counts']['text_supported_target']}` "
                f"rows using a caption heuristic and missed `{manifest['contradiction_target_counts']['text_supported_target_missing']}` rows."
            ),
            f"- Prepared text-target sources: `{json.dumps(contradiction_stats['text_target_source_counts'], sort_keys=True)}`.",
            (
                f"- Among prepared contradiction-row heuristic text targets: `{contradiction_stats['contradiction_diff_from_gold']}` differ from `gold_answer`, "
                f"`{contradiction_stats['contradiction_same_as_gold']}` equal `gold_answer`, and `{contradiction_stats['contradiction_missing']}` are missing."
            ),
            "- Conclusion: raw `different` is not enough to call a row text-contradictory in an answer-grounded sense.",
            "",
            "## Conservative Bottom Line",
            "",
            "- Fully defensible from raw fields alone: vision informativeness, consistent/asymmetric/both-weak pairwise states, and legacy protocol action labels.",
            "- Only partially defensible: text answer-grounded informativeness, pairwise contradiction, modality-conditional targets, and evidential action labels on `different` rows.",
            "- Must be masked for conservative supervision: all raw `caption_state == different` rows when training contradiction-sensitive heads or evidential action targets, plus any irrelevant-modality target slots.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    prepared_jsonl = Path(args.prepared_jsonl)
    prepared_manifest = Path(args.prepared_manifest)
    output_markdown = Path(args.output_markdown)
    output_json = Path(args.output_json)

    manifest = _load_manifest(prepared_manifest)
    raw_schema = _load_raw_schema(args.hf_repo_id, args.hf_revision)
    contradiction_stats = _load_prepared_contradiction_stats(prepared_jsonl)
    target_audit = _build_target_audit(manifest, contradiction_stats)

    payload = {
        "raw_schema": raw_schema,
        "prepared_manifest_summary": {
            "hf_sha": manifest.get("hf_sha"),
            "total_rows_written": manifest.get("total_rows_written"),
            "protocol_category_counts": manifest.get("protocol_category_counts"),
            "contradiction_target_counts": manifest.get("contradiction_target_counts"),
            "drop_counts": manifest.get("drop_counts"),
        },
        "contradiction_prepared_stats": contradiction_stats,
        "target_audit": target_audit,
    }

    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(
        _build_markdown(raw_schema, manifest, contradiction_stats, target_audit),
        encoding="utf-8",
    )
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
