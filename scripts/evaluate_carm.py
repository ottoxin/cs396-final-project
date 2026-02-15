#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from carm.data.io import load_examples
from carm.data.schema import Split
from carm.eval.evaluator import CARMPredictor, evaluate_predictor
from carm.models.backbone import BackboneConfig, MockFrozenBackbone
from carm.models.carm_model import CARMHeads, CARMModelConfig
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CARM predictor.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_ckpt", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    examples = load_examples(args.input_jsonl)
    if args.split != "all":
        wanted = Split(args.split)
        examples = [ex for ex in examples if ex.split == wanted]

    model_cfg = cfg.get("model", {})
    backbone_cfg = cfg.get("backbone", {})
    train_cfg = cfg.get("training", {})

    model = CARMHeads(
        CARMModelConfig(
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            probe_feature_size=int(model_cfg.get("probe_feature_size", 3)),
            pool=str(model_cfg.get("pool", "mean")),
        )
    )

    if args.model_ckpt:
        ckpt = torch.load(args.model_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", {})
        model.load_state_dict(state, strict=False)

    backbone = MockFrozenBackbone(
        BackboneConfig(
            hidden_size=int(backbone_cfg.get("hidden_size", 128)),
            seq_len=int(backbone_cfg.get("seq_len", 32)),
        )
    )

    predictor = CARMPredictor(model=model, backbone=backbone, device=str(train_cfg.get("device", "cpu")))
    metrics = evaluate_predictor(predictor, examples, output_dir=args.output_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
