#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carm.data.io import load_examples
from carm.data.schema import Split
from carm.eval.baselines import (
    BackboneDirectBaseline,
    ProbeOnlyHeuristicBaseline,
    PromptVerificationBaseline,
    TwoPassSelfConsistencyBaseline,
    UncertaintyThresholdAbstainBaseline,
)
from carm.eval.evaluator import evaluate_predictor
from carm.models.registry import create_backbone
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase A baseline evaluators.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--split",
        default="all",
        help="all or comma-separated split names (train,val,test_id,test_ood_family,test_ood_severity,test_ood_hard_swap)",
    )
    return parser.parse_args()


def _parse_split_filter(raw: str) -> set[str] | None:
    if raw.strip().lower() == "all":
        return None
    splits = {s.strip() for s in raw.split(",") if s.strip()}
    valid = {s.value for s in Split}
    unknown = sorted(s for s in splits if s not in valid)
    if unknown:
        raise ValueError(f"Unknown split filters: {unknown}. Valid: {sorted(valid)}")
    return splits


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    examples = load_examples(args.input_jsonl)

    split_filter = _parse_split_filter(args.split)
    if split_filter is not None:
        examples = [ex for ex in examples if ex.split.value in split_filter]

    if not examples:
        raise ValueError("No examples selected for baseline run.")

    backbone = create_backbone(cfg.get("backbone", {}))
    eval_cfg = cfg.get("eval", {})

    baselines = [
        BackboneDirectBaseline(backbone),
        PromptVerificationBaseline(backbone),
        UncertaintyThresholdAbstainBaseline(
            backbone,
            entropy_threshold=float(eval_cfg.get("uncertainty_entropy_threshold", 1.9)),
        ),
        TwoPassSelfConsistencyBaseline(backbone),
        ProbeOnlyHeuristicBaseline(backbone),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for baseline in baselines:
        sub = out_dir / baseline.name
        metrics = evaluate_predictor(baseline, examples, output_dir=sub)
        summary[baseline.name] = metrics

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
