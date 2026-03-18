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

from carm.data.answer_vocab import DEFAULT_COLOR_VOCAB, canonicalize_candidate_answer, normalize_gold_answer
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
from carm.data.schema import Family
from carm.data.vqa_coco import derive_caption_supported_answer

CONTRADICTION_TEXT_SUPPORTED_ANSWER_KEYS = (
    "text_supported_target",
    "c2_text_supported_answer",
    "caption_supported_answer",
    "perturbed_caption_answer",
    "text_answer",
)
HF_SPLIT_TO_INTERNAL = {
    "train": "train",
    "validation": "val",
    "test": "test_id",
}

INFO_STATE_INFORMATIVE = "informative"
INFO_STATE_UNINFORMATIVE = "uninformative"
PAIRWISE_RELATION_CONSISTENT = "consistent"
PAIRWISE_RELATION_CONTRADICTORY = "contradictory"
PAIRWISE_RELATION_ASYMMETRIC = "asymmetric"
PAIRWISE_RELATION_BOTH_WEAK = "both_weak"
JOINT_ANSWER_ABSTAIN = "<ABSTAIN>"
TARGET_DERIVATION_OK = "ok"
TARGET_DERIVATION_PARTIAL = "partial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare nbso/carm-vqa-5way into baseline-ready JSONL with deterministic train/val/test_id splits. "
            "Images are materialized locally under cache_root."
        )
    )
    parser.add_argument("--hf-repo-id", default="nbso/carm-vqa-5way")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument(
        "--hf-split",
        default="all",
        help="HF split to read. Use 'all' to consume official train/validation/test splits when available.",
    )
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
    if str(split).strip().lower() in {"all", "auto", "official"}:
        return load_dataset(repo_id, revision=revision)
    return load_dataset(repo_id, split=split, revision=revision)


def _dataset_mapping_keys(dataset: Any) -> list[str]:
    if not hasattr(dataset, "keys") or not callable(getattr(dataset, "keys")):
        return []
    return [str(key).strip().lower() for key in dataset.keys()]


def _resolve_input_splits(loaded: Any, requested_split: str) -> tuple[dict[str, Any], str]:
    requested = str(requested_split).strip().lower()
    keys = _dataset_mapping_keys(loaded)
    if keys:
        official_keys = [key for key in ("train", "validation", "test") if key in keys]
        if requested in {"all", "auto", "official"}:
            if len(official_keys) >= 2:
                return {key: loaded[key] for key in official_keys}, "hf_official"
            if len(keys) == 1:
                only_key = keys[0]
                return {only_key: loaded[only_key]}, "local_resplit"
            raise ValueError(f"Unsupported HF split layout for request '{requested_split}': {keys}")
        if requested in keys:
            return {requested: loaded[requested]}, "local_resplit"
        raise ValueError(f"Requested HF split '{requested_split}' not found in dataset splits: {keys}")
    return {requested: loaded}, "local_resplit"


def resolve_protocol_oracle_action(source_action: str, protocol_category: str) -> tuple[str, bool]:
    normalized_source = normalize_oracle_action(source_action)
    expected_action = expected_oracle_action_for_category(protocol_category)
    return expected_action, normalized_source != expected_action


def extract_contradiction_text_supported_answer(
    row: dict[str, Any],
    protocol_category: str,
    *,
    question: str,
    family: str,
    caption: str,
) -> tuple[str | None, str | None]:
    if protocol_category != "C4":
        return None, None

    for key in CONTRADICTION_TEXT_SUPPORTED_ANSWER_KEYS:
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


def _family_enum(family: str) -> Family:
    return Family(str(family).strip().lower())


def _canonicalize_supported_target(value: str | None, family: Family) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    canonical = canonicalize_candidate_answer(
        text,
        family,
        recognized_color_labels=DEFAULT_COLOR_VOCAB,
    )
    if canonical is not None:
        return canonical
    return normalize_gold_answer(text, family)


def _info_states_for_category(protocol_category: str) -> tuple[str, str]:
    mapping = {
        "C1": (INFO_STATE_INFORMATIVE, INFO_STATE_INFORMATIVE),
        "C2": (INFO_STATE_INFORMATIVE, INFO_STATE_UNINFORMATIVE),
        "C3": (INFO_STATE_UNINFORMATIVE, INFO_STATE_INFORMATIVE),
        "C4": (INFO_STATE_INFORMATIVE, INFO_STATE_INFORMATIVE),
        "C5": (INFO_STATE_UNINFORMATIVE, INFO_STATE_UNINFORMATIVE),
    }
    out = mapping.get(protocol_category)
    if out is None:
        raise ValueError(f"Unsupported protocol category for info states: {protocol_category}")
    return out


def _pairwise_relation_for_category(protocol_category: str) -> str:
    mapping = {
        "C1": PAIRWISE_RELATION_CONSISTENT,
        "C2": PAIRWISE_RELATION_ASYMMETRIC,
        "C3": PAIRWISE_RELATION_ASYMMETRIC,
        "C4": PAIRWISE_RELATION_CONTRADICTORY,
        "C5": PAIRWISE_RELATION_BOTH_WEAK,
    }
    out = mapping.get(protocol_category)
    if out is None:
        raise ValueError(f"Unsupported protocol category for pairwise relation: {protocol_category}")
    return out


def _joint_answer_for_category(protocol_category: str, gold_answer: str) -> str:
    if protocol_category in {"C4", "C5"}:
        return JOINT_ANSWER_ABSTAIN
    return gold_answer


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

    loaded = _load_hf_rows(args.hf_repo_id, args.hf_revision, args.hf_split)
    source_datasets, split_assignment_mode = _resolve_input_splits(loaded, args.hf_split)
    source_split_available_counts = {name: len(dataset) for name, dataset in source_datasets.items()}
    source_split_read_counts: Counter[str] = Counter()
    total_rows_available = int(sum(source_split_available_counts.values()))
    total_rows_read = 0

    prepared: list[dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    oracle_action_rewrite_count = 0
    contradiction_target_counts: Counter[str] = Counter()
    missing_contradiction_examples: list[str] = []
    noncontradictory_examples: list[str] = []
    contradiction_text_target_source_counts: Counter[str] = Counter()
    label_derivation_status_counts: Counter[str] = Counter()
    pairwise_relation_counts: Counter[str] = Counter()
    vision_info_state_counts: Counter[str] = Counter()
    text_info_state_counts: Counter[str] = Counter()
    joint_answer_counts: Counter[str] = Counter()

    remaining_rows = args.max_rows
    for source_split, dataset in source_datasets.items():
        if remaining_rows is not None and remaining_rows <= 0:
            break
        take_count = len(dataset) if remaining_rows is None else min(len(dataset), int(remaining_rows))
        source_split_read_counts[source_split] = int(take_count)
        if take_count <= 0:
            continue
        rows_iter = dataset if take_count == len(dataset) else dataset.select(range(take_count))
        total_rows_read += int(take_count)
        if remaining_rows is not None:
            remaining_rows -= int(take_count)

        for row in rows_iter:
            try:
                example_id = str(row["example_id"])
                family_raw = str(row["question_family"]).strip().lower()
                family = FAMILY_MAP.get(family_raw)
                if family is None:
                    raise ValueError(f"Unsupported family: {family_raw}")
                family_enum = _family_enum(family)

                raw_oracle_action = str(row["oracle_action"]).strip()
                source_action = normalize_oracle_action(raw_oracle_action)
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
                normalized_gold_answer = _canonicalize_supported_target(str(row["gold_answer"]), family_enum)
                if normalized_gold_answer is None:
                    normalized_gold_answer = str(row["gold_answer"]).strip()

                if protocol_category == "C4":
                    contradiction_target_counts["contradiction_rows"] += 1
                contradiction_text_supported_answer, contradiction_text_target_source = extract_contradiction_text_supported_answer(
                    row,
                    protocol_category,
                    question=str(row["question"]),
                    family=family,
                    caption=text_input,
                )
                contradiction_text_supported_answer = _canonicalize_supported_target(
                    contradiction_text_supported_answer,
                    family_enum,
                )

                vision_info_state, text_info_state = _info_states_for_category(protocol_category)
                pairwise_relation = _pairwise_relation_for_category(protocol_category)
                joint_answer = _joint_answer_for_category(protocol_category, normalized_gold_answer)
                vision_supported_target: str | None = None
                text_supported_target: str | None = None
                vision_target_source = "masked"
                text_target_source = "masked"
                target_derivation_status = TARGET_DERIVATION_OK
                contradiction_supervision_available = False
                target_mask_reason: str | None = None

                if protocol_category == "C1":
                    vision_supported_target = normalized_gold_answer
                    text_supported_target = normalized_gold_answer
                    vision_target_source = "gold_answer"
                    text_target_source = "gold_answer"
                elif protocol_category == "C2":
                    vision_supported_target = normalized_gold_answer
                    vision_target_source = "gold_answer"
                    text_target_source = "masked_uninformative_text"
                elif protocol_category == "C3":
                    text_supported_target = normalized_gold_answer
                    vision_target_source = "masked_uninformative_vision"
                    text_target_source = "gold_answer"
                elif protocol_category == "C4":
                    vision_supported_target = normalized_gold_answer
                    text_supported_target = contradiction_text_supported_answer
                    vision_target_source = "gold_answer"
                    text_target_source = contradiction_text_target_source or "missing_after_caption_rule"
                    if vision_supported_target:
                        contradiction_target_counts["vision_supported_target"] += 1
                    if text_supported_target:
                        contradiction_target_counts["text_supported_target"] += 1
                    else:
                        contradiction_target_counts["text_supported_target_missing"] += 1
                        target_derivation_status = TARGET_DERIVATION_PARTIAL
                        target_mask_reason = "missing_text_supported_target"
                        missing_contradiction_examples.append(example_id)
                    if contradiction_text_target_source is not None:
                        contradiction_text_target_source_counts[contradiction_text_target_source] += 1
                    if text_supported_target and vision_supported_target and text_supported_target != vision_supported_target:
                        contradiction_supervision_available = True
                        contradiction_target_counts["contradiction_validated"] += 1
                    elif text_supported_target and vision_supported_target:
                        contradiction_target_counts["contradiction_not_validated_targets_agree"] += 1
                        target_derivation_status = TARGET_DERIVATION_PARTIAL
                        target_mask_reason = "targets_agree_after_canonicalization"
                        noncontradictory_examples.append(example_id)
                elif protocol_category == "C5":
                    vision_target_source = "masked_uninformative_vision"
                    text_target_source = "masked_uninformative_text"

                image_name = f"{_safe_name(example_id)}.jpg"
                image_path = image_dir / image_name
                if not image_path.exists():
                    image_obj = _as_pil_image(row.get("image") or row.get("image_path"))
                    image_obj.convert("RGB").save(image_path, format="JPEG", quality=int(args.jpeg_quality))

                if "::" in example_id:
                    base_id, variant_id = example_id.split("::", 1)
                else:
                    base_id, variant_id = example_id, protocol_category.lower()

                answer_type = answer_type_for_family(family)
                internal_split = "train"
                if split_assignment_mode == "hf_official":
                    mapped = HF_SPLIT_TO_INTERNAL.get(source_split)
                    if mapped is None:
                        raise ValueError(f"Unsupported official HF split for mapping: {source_split}")
                    internal_split = mapped

                metadata = {
                    "hf_category": str(row.get("category", "")),
                    "protocol_category": protocol_category,
                    "raw_oracle_action": raw_oracle_action,
                    "image_state": str(row.get("image_state", "")),
                    "caption_state": str(row.get("caption_state", "")),
                    "clean_caption": str(row.get("clean_caption", "")),
                    "perturbed_caption": row.get("perturbed_caption"),
                    "vision_target_source": vision_target_source,
                    "text_target_source": text_target_source,
                    "target_derivation_status": target_derivation_status,
                    "contradiction_supervision_available": contradiction_supervision_available,
                    "hf_repo_id": args.hf_repo_id,
                    "hf_revision": args.hf_revision,
                    "hf_source_split": source_split,
                }
                if target_mask_reason is not None:
                    metadata["target_mask_reason"] = target_mask_reason
                if contradiction_text_target_source is not None:
                    metadata["text_supported_target_source"] = contradiction_text_target_source

                prepared.append(
                    {
                        "example_id": example_id,
                        "base_id": base_id,
                        "variant_id": variant_id,
                        "image_path": str(image_path),
                        "text_input": text_input,
                        "question": str(row["question"]),
                        "gold_answer": normalized_gold_answer,
                        "split": internal_split,
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
                        "text_supported_target": text_supported_target,
                        "vision_info_state": vision_info_state,
                        "text_info_state": text_info_state,
                        "pairwise_relation": pairwise_relation,
                        "joint_answer": joint_answer,
                        "metadata": metadata,
                        "record_version": "v1",
                        "protocol_category": protocol_category,
                    }
                )
                family_counts[family] += 1
                category_counts[protocol_category] += 1
                label_derivation_status_counts[target_derivation_status] += 1
                pairwise_relation_counts[pairwise_relation] += 1
                vision_info_state_counts[vision_info_state] += 1
                text_info_state_counts[text_info_state] += 1
                joint_answer_counts[joint_answer] += 1
            except ValueError as exc:
                msg = str(exc).lower()
                if "perturbed caption" in msg:
                    drop_counts["moderation_or_filter_failure"] += 1
                else:
                    drop_counts["malformed"] += 1
            except Exception:
                drop_counts["decode_failure"] += 1

    split_counts: Counter[str] = Counter()
    if split_assignment_mode == "hf_official":
        for row in prepared:
            split_counts[str(row["split"])] += 1
            row.pop("protocol_category", None)
    else:
        split_map = assign_splits_by_base(
            prepared,
            seed=int(args.seed),
            ratios=SplitRatios(train=float(args.train_ratio), val=float(args.val_ratio), test=float(args.test_ratio)),
        )

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
        "split_assignment_mode": split_assignment_mode,
        "hf_source_split_available_counts": dict(sorted(source_split_available_counts.items(), key=lambda kv: kv[0])),
        "hf_source_split_read_counts": dict(sorted(source_split_read_counts.items(), key=lambda kv: kv[0])),
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "ratios_applied": bool(split_assignment_mode != "hf_official"),
        "total_rows_read": total_rows_read,
        "total_rows_available": total_rows_available,
        "total_rows_written": len(prepared),
        "drop_counts": dict(sorted(drop_counts.items(), key=lambda kv: kv[0])),
        "family_counts": dict(sorted(family_counts.items(), key=lambda kv: kv[0])),
        "protocol_category_counts": dict(sorted(category_counts.items(), key=lambda kv: kv[0])),
        "oracle_action_rewrite_count": int(oracle_action_rewrite_count),
        "contradiction_target_counts": dict(sorted(contradiction_target_counts.items(), key=lambda kv: kv[0])),
        "contradiction_text_target_source_counts": dict(sorted(contradiction_text_target_source_counts.items(), key=lambda kv: kv[0])),
        "label_derivation_status_counts": dict(sorted(label_derivation_status_counts.items(), key=lambda kv: kv[0])),
        "pairwise_relation_counts": dict(sorted(pairwise_relation_counts.items(), key=lambda kv: kv[0])),
        "vision_info_state_counts": dict(sorted(vision_info_state_counts.items(), key=lambda kv: kv[0])),
        "text_info_state_counts": dict(sorted(text_info_state_counts.items(), key=lambda kv: kv[0])),
        "joint_answer_counts": dict(sorted(joint_answer_counts.items(), key=lambda kv: kv[0])),
        "missing_contradiction_text_supported_target_examples_preview": missing_contradiction_examples[:20],
        "noncontradictory_examples_preview": noncontradictory_examples[:20],
        "split_counts": dict(sorted(split_counts.items(), key=lambda kv: kv[0])),
        "output_jsonl": str(output_jsonl),
        "output_image_dir": str(image_dir),
    }
    contradiction_rows = int(contradiction_target_counts.get("contradiction_rows", 0))
    contradiction_text_rows = int(contradiction_target_counts.get("text_supported_target", 0))
    contradiction_validated_rows = int(contradiction_target_counts.get("contradiction_validated", 0))
    manifest["contradiction_text_target_coverage"] = (
        float(contradiction_text_rows / contradiction_rows) if contradiction_rows else None
    )
    manifest["validated_contradiction_coverage"] = (
        float(contradiction_validated_rows / contradiction_rows) if contradiction_rows else None
    )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in prepared:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest["status"] = (
        "ok_partial_contradiction_targets"
        if missing_contradiction_examples or noncontradictory_examples
        else "ok"
    )
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote prepared dataset: {output_jsonl}")
    print(f"wrote manifest: {manifest_json}")
    print(json.dumps({"written_rows": len(prepared), "drop_counts": manifest["drop_counts"]}, indent=2))


if __name__ == "__main__":
    main()
