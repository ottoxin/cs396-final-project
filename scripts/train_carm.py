#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from carm.data.io import load_examples
from carm.data.schema import Split
from carm.models.backbone import BackboneConfig, MockFrozenBackbone
from carm.models.carm_model import CARMHeads, CARMModelConfig
from carm.train.trainer import CARMTrainer, TrainerConfig
from carm.utils.config import load_yaml_config
from carm.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CARM heads.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    set_global_seed(int(cfg.get("seed", 7)))

    examples = load_examples(args.train_jsonl)
    train_examples = [ex for ex in examples if ex.split == Split.TRAIN]

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
    backbone = MockFrozenBackbone(
        BackboneConfig(
            hidden_size=int(backbone_cfg.get("hidden_size", 128)),
            seq_len=int(backbone_cfg.get("seq_len", 32)),
        )
    )

    trainer = CARMTrainer(
        model=model,
        backbone=backbone,
        config=TrainerConfig(
            batch_size=int(train_cfg.get("batch_size", 4)),
            epochs=int(train_cfg.get("epochs", 2)),
            lr=float(train_cfg.get("lr", 1e-3)),
            lambda_cf=float(train_cfg.get("lambda_cf", 0.5)),
            margin_cf=float(train_cfg.get("margin_cf", 0.2)),
            device=str(train_cfg.get("device", "cpu")),
        ),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = trainer.train(train_examples, output_dir=out)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }
    ckpt_path = out / "carm_heads.pt"
    torch.save(ckpt, ckpt_path)

    with (out / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"trained on {len(train_examples)} train examples")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
