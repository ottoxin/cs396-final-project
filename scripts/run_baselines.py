#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carm.data.io import load_examples
from carm.eval.baselines import (
    BackboneOnlyBaseline,
    ProbeOnlyHeuristicBaseline,
    PromptVerificationBaseline,
    UncertaintyThresholdAbstainBaseline,
)
from carm.eval.evaluator import evaluate_predictor
from carm.models.backbone import BackboneConfig, MockFrozenBackbone
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline evaluators.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    examples = load_examples(args.input_jsonl)

    backbone_cfg = cfg.get("backbone", {})
    eval_cfg = cfg.get("eval", {})
    backbone = MockFrozenBackbone(
        BackboneConfig(
            hidden_size=int(backbone_cfg.get("hidden_size", 128)),
            seq_len=int(backbone_cfg.get("seq_len", 32)),
        )
    )

    baselines = [
        BackboneOnlyBaseline(backbone),
        PromptVerificationBaseline(backbone),
        UncertaintyThresholdAbstainBaseline(
            backbone,
            entropy_threshold=float(eval_cfg.get("uncertainty_entropy_threshold", 1.9)),
        ),
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
