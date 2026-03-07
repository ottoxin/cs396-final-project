#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from carm.data.answer_vocab import canonicalization_mapping_from_family_vocabs, load_family_vocabs
from carm.data.io import load_examples
from carm.data.schema import Split
from carm.eval.evaluator import CARMPredictor, evaluate_predictor
from carm.models.carm_model import CARMHeads, CARMModelConfig
from carm.models.registry import create_backbone
from carm.train.losses import LossConfig
from carm.utils.config import load_yaml_config
from carm.utils.device import resolve_carm_device
from carm.utils.run_metadata import hash_file_contents, hash_jsonable, resolve_git_commit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CARM predictor.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_ckpt", default=None)
    parser.add_argument(
        "--split",
        default="test_id",
        choices=["train", "val", "test_id", "test_ood_family", "test_ood_severity", "test_ood_hard_swap", "all"],
    )
    parser.add_argument("--track", choices=["answer", "policy", "all"], default="all")
    parser.add_argument("--schema-version", default="2.0")
    parser.add_argument(
        "--report-calibration-heuristic",
        action="store_true",
        help="Include heuristic/fixed-confidence rows in calibration metrics.",
    )
    return parser.parse_args()


def _resolve_answer_canonicalization(eval_cfg: dict[str, object], backbone_cfg: dict[str, object]) -> dict[str, object]:
    resolved = dict(eval_cfg.get("answer_canonicalization", {}) or {})
    vocab_path = backbone_cfg.get("family_vocab_path")
    if isinstance(vocab_path, str) and vocab_path.strip():
        resolved.update(canonicalization_mapping_from_family_vocabs(load_family_vocabs(vocab_path)))
    return resolved


def _resolve_diagnostic_validity(
    cfg: dict[str, object],
    ckpt: dict[str, object] | None,
) -> dict[str, bool]:
    source_cfg = cfg
    if ckpt is not None:
        raw_validity = ckpt.get("diagnostic_validity")
        if isinstance(raw_validity, dict):
            return {
                "conflict": bool(raw_validity.get("conflict", True)),
                "reliability": bool(raw_validity.get("reliability", True)),
            }

        enabled_losses = ckpt.get("enabled_losses")
        if isinstance(enabled_losses, dict):
            return {
                "conflict": bool(enabled_losses.get("conflict", False)),
                "reliability": bool(
                    enabled_losses.get("reliability", False)
                    or enabled_losses.get("counterfactual", False)
                ),
            }

        ckpt_cfg = ckpt.get("config")
        if isinstance(ckpt_cfg, dict):
            source_cfg = ckpt_cfg

    loss_cfg = LossConfig.from_mapping(source_cfg.get("loss"), legacy_training=source_cfg.get("training", {}))
    return loss_cfg.diagnostic_validity()


def _resolve_dataset_manifest_hash(cfg: dict[str, object]) -> str | None:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return None
    paths_cfg = data_cfg.get("paths", {})
    if not isinstance(paths_cfg, dict):
        return None
    manifest_path = paths_cfg.get("prepared_manifest_json")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        return None

    candidate = Path(manifest_path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return hash_file_contents(candidate)


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
    eval_cfg = cfg.get("eval", {})
    canonicalization_cfg = _resolve_answer_canonicalization(eval_cfg, backbone_cfg)

    model = CARMHeads(
        CARMModelConfig(
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            probe_feature_size=int(model_cfg.get("probe_feature_size", 3)),
            pool=str(model_cfg.get("pool", "mean")),
        )
    )

    ckpt: dict[str, object] | None = None
    if args.model_ckpt:
        ckpt = torch.load(args.model_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", {})
        model.load_state_dict(state, strict=False)

    backbone = create_backbone(backbone_cfg)
    if getattr(backbone, "name", "") == "llava_next_8b":
        raise RuntimeError("llava_next_8b is not runnable yet for evaluation. Use qwen2_5_vl_7b.")
    resolved_device = resolve_carm_device(train_cfg.get("device"), backbone)
    resolved_config_hash = hash_jsonable(cfg)
    dataset_manifest_hash = _resolve_dataset_manifest_hash(cfg)
    git_commit = resolve_git_commit(PROJECT_ROOT)

    predictor = CARMPredictor(
        model=model,
        backbone=backbone,
        device=resolved_device,
        diagnostic_validity=_resolve_diagnostic_validity(cfg, ckpt),
    )
    metrics = evaluate_predictor(
        predictor,
        examples,
        output_dir=args.output_dir,
        track=args.track,
        schema_version=args.schema_version,
        semantic_match_threshold=float(eval_cfg.get("semantic_match_threshold", 0.82)),
        canonicalization_cfg=canonicalization_cfg,
        include_heuristic_calibration=bool(args.report_calibration_heuristic),
        resolved_config_hash=resolved_config_hash,
        selected_split=args.split,
        dataset_manifest_hash=dataset_manifest_hash,
        git_commit=git_commit,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
