#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.data.io import load_examples, save_examples
from carm.data.schema import Split
from carm.experimental.labels import derive_labels
from carm.experimental.sampling import SmallRunConfig, build_small_run_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a split-preserving subset from a prepared CARM JSONL export.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sampling_strategy", default="protocol_category_family")
    parser.add_argument(
        "--manifest_json",
        default=None,
        help="Optional manifest path. Defaults to <output_jsonl stem>.manifest.json.",
    )
    return parser.parse_args()


def _resolve_requested_count(total: int, fraction: float) -> int:
    if total <= 0:
        return 0
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return total
    return max(1, math.floor(total * fraction))


def _split_source_sizes(examples: list[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        if example.split in (Split.TRAIN, Split.VAL, Split.TEST_ID):
            counts[example.split.value] += 1
    return {split: int(counts.get(split, 0)) for split in ("train", "val", "test_id")}


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    manifest_json = (
        Path(args.manifest_json)
        if args.manifest_json
        else output_jsonl.with_suffix(".manifest.json")
    )

    if not 0.0 < float(args.fraction) <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")

    examples = load_examples(input_jsonl)
    derived_by_id = {example.example_id: derive_labels(example) for example in examples}
    source_sizes = _split_source_sizes(examples)

    cfg = SmallRunConfig(
        max_train_examples=_resolve_requested_count(source_sizes["train"], float(args.fraction)),
        max_val_examples=_resolve_requested_count(source_sizes["val"], float(args.fraction)),
        max_test_examples=_resolve_requested_count(source_sizes["test_id"], float(args.fraction)),
        random_seed=int(args.seed),
        sampling_strategy=str(args.sampling_strategy),
        run_name=f"sampled_{int(round(float(args.fraction) * 100))}pct_seed{int(args.seed)}",
    )
    selected_splits, sampling_manifest = build_small_run_splits(examples, derived_by_id, cfg)

    sampled_examples = (
        list(selected_splits.get("train", []))
        + list(selected_splits.get("val", []))
        + list(selected_splits.get("test_id", []))
    )
    save_examples(output_jsonl, sampled_examples)

    selected_sizes = {split: len(rows) for split, rows in selected_splits.items()}
    actual_fractions = {
        split: (selected_sizes.get(split, 0) / source_sizes[split]) if source_sizes[split] else 0.0
        for split in ("train", "val", "test_id")
    }
    manifest: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "requested_fraction": float(args.fraction),
        "random_seed": int(args.seed),
        "sampling_strategy": str(args.sampling_strategy),
        "source_split_sizes": source_sizes,
        "selected_split_sizes": selected_sizes,
        "actual_fraction_by_split": actual_fractions,
        "total_selected_examples": len(sampled_examples),
        "note": "Only train/val/test_id splits are sampled and written to the subset export.",
        "sampling_manifest": sampling_manifest,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {len(sampled_examples)} examples to {output_jsonl}")
    print(json.dumps(selected_sizes, indent=2, sort_keys=True))
    print(f"manifest: {manifest_json}")


if __name__ == "__main__":
    main()
