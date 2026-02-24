#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carm.data.construction import build_conflict_suite
from carm.data.io import load_examples, save_examples
from carm.data.schema import Family
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical Conflict Suite v1 from base JSONL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--manifest_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    base_examples = load_examples(args.input_jsonl)

    split_cfg = cfg.get("splits", {})
    conflict_cfg = cfg.get("conflict", {})
    data_cfg = cfg.get("data", {})

    held_out_family = Family(str(split_cfg.get("ood_family", Family.ATTRIBUTE_COLOR.value)))
    held_out_severity = int(split_cfg.get("ood_severity", 3))

    suite, manifest = build_conflict_suite(
        base_examples,
        seed=int(cfg.get("seed", 7)),
        held_out_family=held_out_family,
        held_out_severity=held_out_severity,
        split_ratios=split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test_id": 0.15}),
        vision_corruption_type=str(conflict_cfg.get("vision_corruption", {}).get("type", "occlusion")),
        vision_severities=list(conflict_cfg.get("vision_corruption", {}).get("severities", [1, 2, 3])),
        color_vocab=list(data_cfg.get("color_vocab", [])),
        hard_swap_jaccard_min=float(conflict_cfg.get("swap_hard", {}).get("noun_jaccard_min", 0.2)),
        hard_swap_jaccard_max=float(conflict_cfg.get("swap_hard", {}).get("noun_jaccard_max", 0.7)),
        include_both_variants=bool(conflict_cfg.get("include_both", False)),
        enable_ood_hard_swap=bool(split_cfg.get("enable_ood_hard_swap", False)),
        enforce_template_disjointness=bool(split_cfg.get("enforce_template_disjointness", False)),
    )

    save_examples(args.output_jsonl, suite)

    manifest_path = args.manifest_json
    if manifest_path is None:
        manifest_path = str(Path(args.output_jsonl).with_suffix(".manifest.json"))
    with Path(manifest_path).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {len(suite)} examples to {args.output_jsonl}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
