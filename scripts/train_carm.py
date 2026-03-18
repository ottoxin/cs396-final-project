#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from carm.data.answer_vocab import canonicalization_mapping_from_family_vocabs, load_family_vocabs
from carm.data.io import load_examples
from carm.data.schema import Split
from carm.models.carm_model import CARMHeads, CARMModelConfig
from carm.models.registry import create_backbone
from carm.train.losses import LossConfig
from carm.train.trainer import CARMTrainer, TrainerConfig
from carm.utils.config import load_yaml_config
from carm.utils.device import resolve_carm_device
from carm.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CARM heads.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _resolve_answer_canonicalization(eval_cfg: dict[str, object], backbone_cfg: dict[str, object]) -> dict[str, object]:
    resolved = dict(eval_cfg.get("answer_canonicalization", {}) or {})
    vocab_path = backbone_cfg.get("family_vocab_path")
    if isinstance(vocab_path, str) and vocab_path.strip():
        resolved.update(canonicalization_mapping_from_family_vocabs(load_family_vocabs(vocab_path)))
    return resolved


def _resolved_training_config(train_cfg: dict[str, object]) -> dict[str, object]:
    return {
        "batch_size": int(train_cfg.get("batch_size", 4)),
        "epochs": int(train_cfg.get("epochs", 2)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "early_stop_metric": str(train_cfg.get("early_stop_metric", "task_success")),
        "patience": int(train_cfg.get("patience", 2)),
        "device": str(train_cfg.get("device", "auto")),
        "log_every_steps": int(train_cfg.get("log_every_steps", 50)),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    set_global_seed(int(cfg.get("seed", 7)))

    examples = load_examples(args.train_jsonl)
    train_examples = [ex for ex in examples if ex.split == Split.TRAIN]
    val_examples = [ex for ex in examples if ex.split == Split.VAL]

    model_cfg = cfg.get("model", {})
    backbone_cfg = cfg.get("backbone", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("eval", {})
    loss_cfg = LossConfig.from_mapping(cfg.get("loss"), legacy_training=train_cfg)
    resolved_training = _resolved_training_config(train_cfg)
    canonicalization_cfg = _resolve_answer_canonicalization(eval_cfg, backbone_cfg)

    model = CARMHeads(
        CARMModelConfig(
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            probe_feature_size=int(model_cfg.get("probe_feature_size", 3)),
            pool=str(model_cfg.get("pool", "mean")),
        )
    )
    backbone = create_backbone(backbone_cfg)
    if getattr(backbone, "name", "") == "llava_next_8b":
        raise RuntimeError("llava_next_8b is not runnable yet for training. Use qwen2_5_vl_7b.")
    resolved_device = resolve_carm_device(resolved_training["device"], backbone)

    trainer = CARMTrainer(
        model=model,
        backbone=backbone,
        config=TrainerConfig(
            batch_size=int(resolved_training["batch_size"]),
            epochs=int(resolved_training["epochs"]),
            lr=float(resolved_training["lr"]),
            weight_decay=float(resolved_training["weight_decay"]),
            early_stop_metric=str(resolved_training["early_stop_metric"]),
            patience=int(resolved_training["patience"]),
            device=resolved_device,
            log_every_steps=int(resolved_training["log_every_steps"]),
            loss=loss_cfg,
        ),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = trainer.train(
        train_examples,
        val_examples,
        output_dir=out,
        canonicalization_cfg=canonicalization_cfg,
    )

    resolved_cfg = dict(cfg)
    resolved_cfg["training"] = {**resolved_training, "device": resolved_device}
    resolved_cfg["loss"] = loss_cfg.to_dict()

    ckpt = {
        "model_state_dict": result.best_model_state_dict,
        "config": resolved_cfg,
        "metrics": result.best_val_metrics,
        "label_mapping": result.label_mapping,
        "enabled_losses": result.enabled_losses,
        "diagnostic_validity": result.diagnostic_validity,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "early_stop_reason": result.early_stop_reason,
    }
    ckpt_path = out / "carm_heads.pt"
    torch.save(ckpt, ckpt_path)

    with (out / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2)
    with (out / "best_val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result.best_val_metrics, f, indent=2)
    with (out / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "action_to_idx": result.label_mapping,
                "idx_to_action": [label for label, _ in sorted(result.label_mapping.items(), key=lambda item: item[1])],
            },
            f,
            indent=2,
        )

    print(f"trained on {len(train_examples)} train examples")
    print(f"validated on {len(val_examples)} val examples")
    print(f"best epoch: {result.best_epoch}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
