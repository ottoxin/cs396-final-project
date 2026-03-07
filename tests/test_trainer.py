from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from carm.data.io import save_examples
from carm.data.schema import Action, CorruptModality, Split
from carm.eval.evaluator import CARMPredictor, evaluate_predictor
from carm.models.carm_model import CARMHeads, CARMModelConfig
from carm.train.losses import LossConfig, build_targets, multi_task_loss
from carm.utils.device import resolve_carm_device
from tests.dummy_backbone import DeterministicTestBackbone
from tests.fixtures import make_base_examples


def _make_examples() -> list:
    base = make_base_examples()

    def _clone(idx: int, *, example_id: str, split: Split, action: Action, corrupt: CorruptModality):
        ex = copy.deepcopy(base[idx])
        ex.example_id = example_id
        ex.base_id = example_id.split("::")[0]
        ex.variant_id = example_id.split("::")[-1]
        ex.split = split
        ex.oracle_action = action
        ex.corrupt_modality = corrupt
        ex.metadata = {
            "protocol_category": {
                Action.REQUIRE_AGREEMENT: "C1",
                Action.TRUST_VISION: "C3",
                Action.TRUST_TEXT: "C4",
                Action.ABSTAIN: "C5",
            }[action]
        }
        return ex

    return [
        _clone(0, example_id="train-1::clean", split=Split.TRAIN, action=Action.REQUIRE_AGREEMENT, corrupt=CorruptModality.NONE),
        _clone(1, example_id="train-2::vision", split=Split.TRAIN, action=Action.TRUST_TEXT, corrupt=CorruptModality.VISION),
        _clone(2, example_id="train-3::text", split=Split.TRAIN, action=Action.TRUST_VISION, corrupt=CorruptModality.TEXT),
        _clone(0, example_id="train-4::both", split=Split.TRAIN, action=Action.ABSTAIN, corrupt=CorruptModality.BOTH),
        _clone(1, example_id="val-1::clean", split=Split.VAL, action=Action.REQUIRE_AGREEMENT, corrupt=CorruptModality.NONE),
        _clone(2, example_id="val-2::vision", split=Split.VAL, action=Action.TRUST_TEXT, corrupt=CorruptModality.VISION),
        _clone(0, example_id="val-3::text", split=Split.VAL, action=Action.TRUST_VISION, corrupt=CorruptModality.TEXT),
        _clone(1, example_id="val-4::both", split=Split.VAL, action=Action.ABSTAIN, corrupt=CorruptModality.BOTH),
    ]


class TestTrainingLosses(unittest.TestCase):
    def test_loss_config_uses_legacy_multitask_defaults_when_loss_block_is_missing(self) -> None:
        cfg = LossConfig.from_mapping(None, legacy_training={"lambda_cf": 0.5, "margin_cf": 0.2})

        self.assertTrue(cfg.action)
        self.assertTrue(cfg.conflict)
        self.assertTrue(cfg.reliability)
        self.assertTrue(cfg.counterfactual)
        self.assertEqual(cfg.lambda_conf, 1.0)
        self.assertEqual(cfg.lambda_rel, 1.0)
        self.assertEqual(cfg.lambda_cf, 0.5)
        self.assertEqual(cfg.margin_cf, 0.2)

    def test_action_only_loss_uses_only_action_path(self) -> None:
        example = _make_examples()[0]
        targets = build_targets(example, device=torch.device("cpu"))
        conflict_logits = torch.randn(1, 4, requires_grad=True)
        action_logits = torch.randn(1, 4, requires_grad=True)
        reliability_pred = torch.randn(1, 2, requires_grad=True)
        cf_loss = torch.tensor(0.7, requires_grad=True)

        total, logs = multi_task_loss(
            conflict_logits=conflict_logits,
            action_logits=action_logits,
            reliability_pred=reliability_pred,
            targets=targets,
            cf_loss=cf_loss,
            loss_cfg=LossConfig(action=True, conflict=False, reliability=False, counterfactual=False),
        )

        grads = torch.autograd.grad(total, [conflict_logits, action_logits, reliability_pred, cf_loss], allow_unused=True)
        self.assertIsNone(grads[0])
        self.assertIsNotNone(grads[1])
        self.assertIsNone(grads[2])
        self.assertIsNone(grads[3])
        self.assertGreater(logs["loss_action"], 0.0)
        self.assertEqual(logs["loss_conflict"], 0.0)
        self.assertEqual(logs["loss_reliability"], 0.0)
        self.assertEqual(logs["loss_cf"], 0.0)

    def test_auxiliary_losses_contribute_when_enabled(self) -> None:
        example = _make_examples()[0]
        targets = build_targets(example, device=torch.device("cpu"))
        conflict_logits = torch.randn(1, 4, requires_grad=True)
        action_logits = torch.randn(1, 4, requires_grad=True)
        reliability_pred = torch.randn(1, 2, requires_grad=True)
        cf_loss = torch.tensor(0.7, requires_grad=True)

        total, _ = multi_task_loss(
            conflict_logits=conflict_logits,
            action_logits=action_logits,
            reliability_pred=reliability_pred,
            targets=targets,
            cf_loss=cf_loss,
            loss_cfg=LossConfig(
                action=True,
                conflict=True,
                reliability=True,
                counterfactual=True,
                lambda_conf=1.0,
                lambda_rel=1.0,
                lambda_cf=0.5,
            ),
        )

        grads = torch.autograd.grad(total, [conflict_logits, action_logits, reliability_pred, cf_loss], allow_unused=True)
        self.assertIsNotNone(grads[0])
        self.assertIsNotNone(grads[1])
        self.assertIsNotNone(grads[2])
        self.assertIsNotNone(grads[3])


class TestDeviceResolution(unittest.TestCase):
    class _BackboneWithDevice:
        def __init__(self, device) -> None:
            self.device = device

    def test_explicit_training_device_overrides_backbone_device(self) -> None:
        self.assertEqual(resolve_carm_device("cpu", self._BackboneWithDevice(torch.device("cuda"))), "cpu")
        self.assertEqual(resolve_carm_device("cuda", self._BackboneWithDevice(torch.device("cpu"))), "cuda")

    def test_auto_training_device_follows_backbone_device(self) -> None:
        self.assertEqual(resolve_carm_device("auto", self._BackboneWithDevice(torch.device("cpu"))), "cpu")
        self.assertEqual(resolve_carm_device(None, self._BackboneWithDevice("cuda")), "cuda")

    def test_resolve_carm_device_falls_back_to_cpu_without_backbone_device(self) -> None:
        self.assertEqual(resolve_carm_device("auto", object()), "cpu")


class TestTrainingScripts(unittest.TestCase):
    def test_training_and_eval_scripts_write_best_checkpoint_and_null_aux_diagnostics(self) -> None:
        from scripts import evaluate_carm, train_carm

        examples = _make_examples()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_path = root / "examples.jsonl"
            cfg_path = root / "config.yaml"
            train_out = root / "train_out"
            eval_out = root / "eval_out"

            save_examples(data_path, examples)
            cfg_path.write_text(
                "\n".join(
                    [
                        "seed: 7",
                        "model:",
                        "  hidden_size: 128",
                        "  probe_feature_size: 3",
                        "  pool: mean",
                        "backbone:",
                        "  name: qwen2_5_vl_7b",
                        "training:",
                        "  batch_size: 2",
                        "  epochs: 3",
                        "  lr: 0.001",
                        "  weight_decay: 0.01",
                        "  early_stop_metric: task_success",
                        "  patience: 1",
                        "  device: auto",
                        "loss:",
                        "  action: true",
                        "  conflict: false",
                        "  reliability: false",
                        "  counterfactual: false",
                        "  lambda_conf: 0.0",
                        "  lambda_rel: 0.0",
                        "  lambda_cf: 0.0",
                        "  margin_cf: 0.2",
                        "eval: {}",
                    ]
                ),
                encoding="utf-8",
            )

            train_backbone = DeterministicTestBackbone()
            train_backbone.device = torch.device("cpu")
            with patch.object(train_carm, "create_backbone", return_value=train_backbone):
                with patch.object(
                    sys,
                    "argv",
                    ["train_carm.py", "--config", str(cfg_path), "--train_jsonl", str(data_path), "--output_dir", str(train_out)],
                ):
                    with redirect_stdout(io.StringIO()):
                        train_carm.main()

            for name in (
                "carm_heads.pt",
                "resolved_config.json",
                "best_val_metrics.json",
                "train_history.jsonl",
                "label_mapping.json",
                "val_predictions_best.jsonl",
            ):
                self.assertTrue((train_out / name).exists(), name)

            best_metrics = json.loads((train_out / "best_val_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("best_epoch", best_metrics)
            self.assertIn("stopped_epoch", best_metrics)
            self.assertIn("early_stop_reason", best_metrics)
            self.assertIn("task_success", best_metrics)
            self.assertIn("action_accuracy", best_metrics)
            self.assertIn("action_macro_f1", best_metrics)
            resolved_cfg = json.loads((train_out / "resolved_config.json").read_text(encoding="utf-8"))
            self.assertEqual(resolved_cfg["training"]["device"], "cpu")

            ckpt = torch.load(train_out / "carm_heads.pt", map_location="cpu")
            self.assertEqual(ckpt["enabled_losses"], {"action": True, "conflict": False, "reliability": False, "counterfactual": False})
            self.assertEqual(ckpt["diagnostic_validity"], {"conflict": False, "reliability": False})

            eval_backbone = DeterministicTestBackbone()
            eval_backbone.device = torch.device("cpu")
            with patch.object(evaluate_carm, "create_backbone", return_value=eval_backbone):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_carm.py",
                        "--config",
                        str(cfg_path),
                        "--input_jsonl",
                        str(data_path),
                        "--output_dir",
                        str(eval_out),
                        "--model_ckpt",
                        str(train_out / "carm_heads.pt"),
                        "--split",
                        "val",
                    ],
                ):
                    with redirect_stdout(io.StringIO()):
                        evaluate_carm.main()

            metrics = json.loads((eval_out / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("action_accuracy", metrics)
            self.assertIn("action_macro_f1", metrics)

            row = json.loads((eval_out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNotNone(row["pred_action"])
            self.assertIsNone(row["pred_conflict_type"])
            self.assertIsNone(row["r_v"])
            self.assertIsNone(row["r_t"])
            self.assertIn("c2_vision_only_correct", row)
            self.assertIn("c2_text_only_correct", row)
            self.assertIn("c2_multimodal_abstained", row)

    def test_action_only_carm_predictor_keeps_nullable_aux_fields(self) -> None:
        examples = _make_examples()[:1]
        model = CARMHeads(CARMModelConfig())
        predictor = CARMPredictor(
            model=model,
            backbone=DeterministicTestBackbone(),
            diagnostic_validity={"conflict": False, "reliability": False},
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            evaluate_predictor(predictor, examples, out, progress_every=0)
            row = json.loads((out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertIn("pred_conflict_type", row)
        self.assertIn("r_v", row)
        self.assertIn("r_t", row)
        self.assertIn("c2_vision_only_correct", row)
        self.assertIn("c2_text_only_correct", row)
        self.assertIn("c2_multimodal_abstained", row)
        self.assertIsNone(row["pred_conflict_type"])
        self.assertIsNone(row["r_v"])
        self.assertIsNone(row["r_t"])
        self.assertIsNone(row["c2_vision_only_correct"])
        self.assertIsNone(row["c2_text_only_correct"])
        self.assertIsNone(row["c2_multimodal_abstained"])


if __name__ == "__main__":
    unittest.main()
