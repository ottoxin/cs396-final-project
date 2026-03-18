from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from carm.data.schema import ConflictExample, CorruptModality
from carm.eval.evaluator import PREDICTIONS_FILENAME, CARMPredictor, evaluate_predictor
from carm.train.losses import ACTION_TO_IDX, LossConfig
from carm.models.interfaces import BackboneAdapter
from carm.models.carm_model import CARMHeads
from carm.train.dataset import ConflictDataset, build_clean_index, pair_key
from carm.train.losses import (
    build_targets,
    counterfactual_hinge,
    multi_task_loss,
)


@dataclass
class TrainerConfig:
    batch_size: int = 4
    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.01
    early_stop_metric: str = "task_success"
    patience: int = 2
    device: str = "cpu"
    log_every_steps: int = 50
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class TrainingResult:
    best_model_state_dict: dict[str, torch.Tensor]
    best_val_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    label_mapping: dict[str, int]
    enabled_losses: dict[str, bool]
    diagnostic_validity: dict[str, bool]
    best_epoch: int
    stopped_epoch: int
    early_stop_reason: str


class CARMTrainer:
    def __init__(
        self,
        model: CARMHeads,
        backbone: BackboneAdapter,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.backbone = backbone
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _clear_backbone_caches(self) -> None:
        clear_fn = getattr(self.backbone, "clear_caches", None)
        if callable(clear_fn):
            clear_fn()

    def _forward_example(self, ex: ConflictExample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_payload = ex.image_path
        recipe = ex.metadata.get("vision_recipe") if isinstance(ex.metadata, dict) else None
        if isinstance(recipe, dict) and "payload" in recipe:
            image_payload = str(recipe["payload"])

        with torch.no_grad():
            mm = self.backbone.run_backbone_multimodal(image_payload, ex.text_input, ex.question)
            pv = self.backbone.run_probe_vision_only(image_payload, ex.question)
            pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        anchor_states = mm.hidden_states.to(self.device)
        phi_v = pv.features.to(self.device)
        phi_t = pt.features.to(self.device)
        conflict_logits, reliability, action_logits = self.model.carm_forward(anchor_states, phi_v, phi_t)
        return conflict_logits, reliability, action_logits

    @staticmethod
    def _state_dict_to_cpu(model: CARMHeads) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    @staticmethod
    def _history_record(
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, Any],
        *,
        is_best: bool,
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "is_best": bool(is_best),
            "train_loss_total": float(train_metrics.get("loss_total", 0.0)),
            "train_loss_action": float(train_metrics.get("loss_action", 0.0)),
            "train_loss_conflict": float(train_metrics.get("loss_conflict", 0.0)),
            "train_loss_reliability": float(train_metrics.get("loss_reliability", 0.0)),
            "train_loss_cf": float(train_metrics.get("loss_cf", 0.0)),
            "val_task_success": val_metrics.get("task_success"),
            "val_action_accuracy": val_metrics.get("action_accuracy"),
            "val_action_macro_f1": val_metrics.get("action_macro_f1"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_coverage": val_metrics.get("coverage"),
            "val_accuracy_on_answered": val_metrics.get("accuracy_on_answered"),
        }

    def _is_improved(
        self,
        metrics: dict[str, Any],
        best_metrics: dict[str, Any] | None,
    ) -> bool:
        if best_metrics is None:
            return True

        metric_key = str(self.config.early_stop_metric)
        current_primary = float(metrics.get(metric_key, float("-inf")) or float("-inf"))
        best_primary = float(best_metrics.get(metric_key, float("-inf")) or float("-inf"))
        if current_primary > best_primary:
            return True
        if current_primary < best_primary:
            return False

        current_action_acc = float(metrics.get("action_accuracy", float("-inf")) or float("-inf"))
        best_action_acc = float(best_metrics.get("action_accuracy", float("-inf")) or float("-inf"))
        return current_action_acc > best_action_acc

    def _train_epoch(
        self,
        epoch: int,
        loader: DataLoader[list[ConflictExample]],
        clean_index: dict[str, ConflictExample],
        *,
        progress_file: Any | None = None,
    ) -> dict[str, float]:
        metrics_accum: dict[str, float] = {
            "loss_total": 0.0,
            "loss_conflict": 0.0,
            "loss_action": 0.0,
            "loss_reliability": 0.0,
            "loss_cf": 0.0,
        }
        examples_seen = 0
        steps_done = 0
        total_steps = len(loader)
        log_every_steps = max(0, int(self.config.log_every_steps))
        start_time = time.perf_counter()

        self.model.train()
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.tensor(0.0, device=self.device)
            batch_logs: list[dict[str, float]] = []

            for ex in batch:
                conflict_logits, reliability, action_logits = self._forward_example(ex)
                targets = build_targets(ex, device=self.device)

                cf = torch.tensor(0.0, device=self.device)
                if self.config.loss.counterfactual and ex.corrupt_modality != CorruptModality.NONE:
                    ref = clean_index.get(pair_key(ex))
                    if ref is not None:
                        _, clean_rel, _ = self._forward_example(ref)
                        cf = counterfactual_hinge(
                            clean_reliability=clean_rel.squeeze(0),
                            corrupted_reliability=reliability.squeeze(0),
                            corrupted_modality=ex.corrupt_modality,
                            margin=self.config.loss.margin_cf,
                        )

                total, logs = multi_task_loss(
                    conflict_logits=conflict_logits,
                    action_logits=action_logits,
                    reliability_pred=reliability,
                    targets=targets,
                    cf_loss=cf,
                    loss_cfg=self.config.loss,
                )
                batch_loss = batch_loss + total
                batch_logs.append(logs)
                examples_seen += 1

            batch_loss.backward()
            self.optimizer.step()
            steps_done += 1

            for logs in batch_logs:
                for key, value in logs.items():
                    metrics_accum[key] += value

            should_log = log_every_steps > 0 and (steps_done % log_every_steps == 0 or steps_done == total_steps)
            if should_log:
                elapsed_sec = max(1e-9, time.perf_counter() - start_time)
                avg_step_sec = elapsed_sec / max(1, steps_done)
                eta_sec = max(0.0, (total_steps - steps_done) * avg_step_sec)
                record = {
                    "phase": "train_progress",
                    "epoch": int(epoch),
                    "step": int(steps_done),
                    "total_steps": int(total_steps),
                    "examples_seen": int(examples_seen),
                    "pct_complete": float(steps_done / max(1, total_steps)),
                    "elapsed_sec": float(elapsed_sec),
                    "eta_sec": float(eta_sec),
                    "examples_per_sec": float(examples_seen / elapsed_sec),
                    "avg_loss_total": float(metrics_accum["loss_total"] / max(1, examples_seen)),
                    "avg_loss_action": float(metrics_accum["loss_action"] / max(1, examples_seen)),
                    "avg_loss_conflict": float(metrics_accum["loss_conflict"] / max(1, examples_seen)),
                    "avg_loss_reliability": float(metrics_accum["loss_reliability"] / max(1, examples_seen)),
                    "avg_loss_cf": float(metrics_accum["loss_cf"] / max(1, examples_seen)),
                }
                line = json.dumps(record, ensure_ascii=True)
                print(line, flush=True)
                if progress_file is not None:
                    progress_file.write(line + "\n")
                    progress_file.flush()

        return {
            key: (value / max(1, examples_seen))
            for key, value in metrics_accum.items()
        }

    def _validate(
        self,
        examples: list[ConflictExample],
        *,
        output_dir: Path,
        canonicalization_cfg: Any,
    ) -> dict[str, Any]:
        predictor = CARMPredictor(
            model=self.model,
            backbone=self.backbone,
            device=str(self.device),
            diagnostic_validity=self.config.loss.diagnostic_validity(),
        )
        return evaluate_predictor(
            predictor,
            examples,
            output_dir=output_dir,
            progress_every=0,
            log_fn=lambda _: None,
            canonicalization_cfg=canonicalization_cfg,
        )

    def train(
        self,
        train_examples: list[ConflictExample],
        val_examples: list[ConflictExample],
        output_dir: str | Path,
        *,
        canonicalization_cfg: Any = None,
    ) -> TrainingResult:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not train_examples:
            raise ValueError("CARM training requires at least one train example.")
        if not val_examples:
            raise ValueError("CARM training requires at least one validation example.")

        dataset = ConflictDataset(train_examples)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=list)
        clean_index = build_clean_index(train_examples)

        history: list[dict[str, Any]] = []
        best_metrics: dict[str, Any] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        stopped_epoch = 0
        early_stop_reason = "completed_all_epochs"
        epochs_without_improvement = 0

        history_path = out_dir / "train_history.jsonl"
        progress_path = out_dir / "train_progress.jsonl"
        best_predictions_path = out_dir / "val_predictions_best.jsonl"
        best_metrics_path = out_dir / "best_val_metrics.json"

        with history_path.open("w", encoding="utf-8") as history_file, progress_path.open("w", encoding="utf-8") as progress_file:
            for epoch in range(1, self.config.epochs + 1):
                train_metrics = self._train_epoch(epoch, loader, clean_index, progress_file=progress_file)
                self._clear_backbone_caches()

                with tempfile.TemporaryDirectory(dir=out_dir) as td:
                    val_dir = Path(td) / "val_eval"
                    val_metrics = self._validate(
                        val_examples,
                        output_dir=val_dir,
                        canonicalization_cfg=canonicalization_cfg,
                    )
                    self._clear_backbone_caches()

                    improved = self._is_improved(val_metrics, best_metrics)
                    record = self._history_record(epoch, train_metrics, val_metrics, is_best=improved)
                    history.append(record)
                    history_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                    history_file.flush()
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "train_loss_total": record["train_loss_total"],
                                "val_task_success": record["val_task_success"],
                                "val_action_accuracy": record["val_action_accuracy"],
                                "is_best": improved,
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )

                    if improved:
                        best_metrics = dict(val_metrics)
                        best_epoch = epoch
                        best_state_dict = self._state_dict_to_cpu(self.model)
                        epochs_without_improvement = 0
                        shutil.copyfile(val_dir / PREDICTIONS_FILENAME, best_predictions_path)
                    else:
                        epochs_without_improvement += 1

                if best_metrics is not None:
                    best_metrics_payload = {
                        **best_metrics,
                        "best_epoch": best_epoch,
                        "stopped_epoch": epoch,
                        "early_stop_reason": early_stop_reason,
                    }
                    with best_metrics_path.open("w", encoding="utf-8") as f:
                        json.dump(best_metrics_payload, f, indent=2)

                if epochs_without_improvement >= self.config.patience:
                    stopped_epoch = epoch
                    early_stop_reason = (
                        f"no_improvement_for_{self.config.patience}_epochs_on_{self.config.early_stop_metric}"
                    )
                    break
            else:
                stopped_epoch = self.config.epochs

        if best_state_dict is None or best_metrics is None:
            raise RuntimeError("Training completed without selecting a best checkpoint.")

        self.model.load_state_dict(best_state_dict, strict=True)
        best_metrics_payload = {
            **best_metrics,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "early_stop_reason": early_stop_reason,
        }
        with best_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(best_metrics_payload, f, indent=2)

        return TrainingResult(
            best_model_state_dict=best_state_dict,
            best_val_metrics=best_metrics_payload,
            history=history,
            label_mapping={action.value: idx for action, idx in ACTION_TO_IDX.items()},
            enabled_losses=self.config.loss.enabled_losses(),
            diagnostic_validity=self.config.loss.diagnostic_validity(),
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            early_stop_reason=early_stop_reason,
        )
