#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from carm.data.hf5way import (
    FAMILY_MAP,
    SplitRatios,
    answer_type_for_family,
    assign_splits_by_base,
    choose_text_input,
    derive_protocol_category,
    expected_oracle_action_for_category,
    normalize_oracle_action,
    schema_fields_for_category,
)
from carm.data.vqa_coco import derive_caption_supported_answer

C2_TEXT_SUPPORTED_ANSWER_KEYS = (
    "text_supported_target",
    "c2_text_supported_answer",
    "caption_supported_answer",
    "perturbed_caption_answer",
    "text_answer",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare nbso/carm-vqa-5way into baseline-ready JSONL with deterministic train/val/test_id splits. "
            "Images are materialized locally under cache_root."
        )
    )
    parser.add_argument("--hf-repo-id", default="nbso/carm-vqa-5way")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--cache-root", default="data/cache/hf_5way")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _as_pil_image(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, dict):
        if item.get("bytes") is not None:
            return Image.open(io.BytesIO(item["bytes"]))
        if item.get("path"):
            return Image.open(item["path"])
    raise ValueError("Unsupported HF image payload.")


def _resolve_sha(repo_id: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi
    except Exception:
        return "unknown"
    try:
        info = HfApi().dataset_info(repo_id, revision=revision)
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha:
            return sha
    except Exception:
        return "unknown"
    return "unknown"


def _load_hf_rows(repo_id: str, revision: str, split: str):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc
    return load_dataset(repo_id, split=split, revision=revision)


def resolve_protocol_oracle_action(source_action: str, protocol_category: str) -> tuple[str, bool]:
    normalized_source = normalize_oracle_action(source_action)
    expected_action = expected_oracle_action_for_category(protocol_category)
    return expected_action, normalized_source != expected_action


def extract_c2_text_supported_answer(
    row: dict[str, Any],
    protocol_category: str,
    *,
    question: str,
    family: str,
    caption: str,
) -> tuple[str | None, str | None]:
    if protocol_category != "C2":
        return None, None

    for key in C2_TEXT_SUPPORTED_ANSWER_KEYS:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, f"explicit:{key}"

    derived = derive_caption_supported_answer(question, family, caption)
    if derived is not None:
        return derived, "derived_from_caption_rule"
    return None, "missing_after_caption_rule"


def main() -> None:
    args = parse_args()

    cache_root = Path(args.cache_root)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else cache_root / "prepared" / "carm_vqa_5way.jsonl"
    manifest_json = (
        Path(args.manifest_json)
        if args.manifest_json
        else cache_root / "prepared" / "carm_vqa_5way.manifest.json"
    )
    image_dir = Path(args.image_dir) if args.image_dir else cache_root / "images"

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_hf_rows(args.hf_repo_id, args.hf_revision, args.hf_split)
    total_rows = len(dataset)
    rows_iter = dataset if args.max_rows is None else dataset.select(range(min(args.max_rows, total_rows)))

    prepared: list[dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    oracle_action_rewrite_count = 0
    c2_target_counts: Counter[str] = Counter()
    missing_c2_examples: list[str] = []
    c2_text_target_source_counts: Counter[str] = Counter()

    for row in rows_iter:
        try:
            example_id = str(row["example_id"])
            family_raw = str(row["question_family"]).strip().lower()
            family = FAMILY_MAP.get(family_raw)
            if family is None:
                raise ValueError(f"Unsupported family: {family_raw}")

            source_action = normalize_oracle_action(str(row["oracle_action"]))
            protocol_category = derive_protocol_category(
                image_state=str(row["image_state"]),
                caption_state=str(row["caption_state"]),
                oracle_action=source_action,
            )
            operator, corrupt_modality, severity, expected_action = schema_fields_for_category(protocol_category)
            normalized_action, was_rewritten = resolve_protocol_oracle_action(source_action, protocol_category)
            if normalized_action != expected_action:
                raise ValueError(
                    f"Protocol action mismatch: action={normalized_action}, expected={expected_action}, category={protocol_category}"
                )
            if was_rewritten:
                oracle_action_rewrite_count += 1

            text_input = choose_text_input(
                caption_state=str(row["caption_state"]),
                clean_caption=str(row["clean_caption"]),
                perturbed_caption=row.get("perturbed_caption"),
            )
            if protocol_category == "C2":
                c2_target_counts["c2_rows"] += 1
            c2_text_supported_answer, c2_text_target_source = extract_c2_text_supported_answer(
                row,
                protocol_category,
                question=str(row["question"]),
                family=family,
                caption=text_input,
            )
            vision_supported_target = None
            if protocol_category == "C2":
                vision_supported_target = str(row["gold_answer"]).strip()
                if vision_supported_target:
                    c2_target_counts["vision_supported_target"] += 1
                if c2_text_supported_answer:
                    c2_target_counts["text_supported_target"] += 1
                else:
                    c2_target_counts["text_supported_target_missing"] += 1
                    missing_c2_examples.append(example_id)
                if c2_text_target_source is not None:
                    c2_text_target_source_counts[c2_text_target_source] += 1

            image_obj = _as_pil_image(row["image_path"])
            image_name = f"{_safe_name(example_id)}.jpg"
            image_path = image_dir / image_name
            image_obj.convert("RGB").save(image_path, format="JPEG", quality=int(args.jpeg_quality))

            if "::" in example_id:
                base_id, variant_id = example_id.split("::", 1)
            else:
                base_id, variant_id = example_id, protocol_category.lower()

            answer_type = answer_type_for_family(family)
            metadata = {
                "hf_category": str(row.get("category", "")),
                "protocol_category": protocol_category,
                "image_state": str(row.get("image_state", "")),
                "caption_state": str(row.get("caption_state", "")),
                "clean_caption": str(row.get("clean_caption", "")),
                "perturbed_caption": row.get("perturbed_caption"),
                "hf_repo_id": args.hf_repo_id,
                "hf_revision": args.hf_revision,
            }
            if c2_text_target_source is not None:
                metadata["text_supported_target_source"] = c2_text_target_source

            prepared.append(
                {
                    "example_id": example_id,
                    "base_id": base_id,
                    "variant_id": variant_id,
                    "image_path": str(image_path),
                    "text_input": text_input,
                    "question": str(row["question"]),
                    "gold_answer": str(row["gold_answer"]),
                    "split": "train",
                    "family": family,
                    "operator": operator,
                    "corrupt_modality": corrupt_modality,
                    "severity": severity,
                    "answer_type": answer_type,
                    "oracle_action": normalized_action,
                    "source_image_id": base_id,
                    "template_id": None,
                    "evidence_modality": "both",
                    "vision_supported_target": vision_supported_target,
                    "text_supported_target": c2_text_supported_answer,
                    "metadata": metadata,
                    "record_version": "v1",
                    "protocol_category": protocol_category,
                }
            )
            family_counts[family] += 1
            category_counts[protocol_category] += 1
        except ValueError as exc:
            msg = str(exc).lower()
            if "perturbed caption" in msg:
                drop_counts["moderation_or_filter_failure"] += 1
            else:
                drop_counts["malformed"] += 1
        except Exception:
            drop_counts["decode_failure"] += 1

    split_map = assign_splits_by_base(
        prepared,
        seed=int(args.seed),
        ratios=SplitRatios(train=float(args.train_ratio), val=float(args.val_ratio), test=float(args.test_ratio)),
    )

    split_counts: Counter[str] = Counter()
    for row in prepared:
        split = split_map[str(row["base_id"])]
        row["split"] = split
        split_counts[split] += 1
        row.pop("protocol_category", None)

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "hf_repo_id": args.hf_repo_id,
        "hf_revision": args.hf_revision,
        "hf_sha": _resolve_sha(args.hf_repo_id, args.hf_revision),
        "hf_split": args.hf_split,
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "total_rows_read": len(rows_iter),
        "total_rows_available": total_rows,
        "total_rows_written": len(prepared),
        "drop_counts": dict(sorted(drop_counts.items(), key=lambda kv: kv[0])),
        "family_counts": dict(sorted(family_counts.items(), key=lambda kv: kv[0])),
        "protocol_category_counts": dict(sorted(category_counts.items(), key=lambda kv: kv[0])),
        "oracle_action_rewrite_count": int(oracle_action_rewrite_count),
        "c2_target_counts": dict(sorted(c2_target_counts.items(), key=lambda kv: kv[0])),
        "c2_text_target_source_counts": dict(sorted(c2_text_target_source_counts.items(), key=lambda kv: kv[0])),
        "missing_c2_text_supported_target_examples_preview": missing_c2_examples[:20],
        "split_counts": dict(sorted(split_counts.items(), key=lambda kv: kv[0])),
        "output_jsonl": str(output_jsonl),
        "output_image_dir": str(image_dir),
    }
    c2_rows = int(c2_target_counts.get("c2_rows", 0))
    c2_text_rows = int(c2_target_counts.get("text_supported_target", 0))
    manifest["c2_text_target_coverage"] = float(c2_text_rows / c2_rows) if c2_rows else None

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in prepared:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest["status"] = "ok_partial_c2_text_targets" if missing_c2_examples else "ok"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote prepared dataset: {output_jsonl}")
    print(f"wrote manifest: {manifest_json}")
    print(json.dumps({"written_rows": len(prepared), "drop_counts": manifest["drop_counts"]}, indent=2))


if __name__ == "__main__":
    main()
