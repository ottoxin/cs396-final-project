from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from carm.data.schema import ConflictExample
from carm.experimental.labels import ACTION_TO_IDX, DerivedLabels, INFO_STATE_TO_IDX, PAIRWISE_RELATION_TO_IDX
from carm.experimental.model import CascadeCARMHeads, DistributionCARMHeads, ExperimentalCARMHeads
from carm.models.features import extract_cross_modal_features
from carm.models.interfaces import BackboneAdapter


@dataclass
class ExperimentalLossConfig:
    lambda_vision_info: float = 1.0
    lambda_text_info: float = 1.0
    lambda_relation: float = 1.0
    lambda_action: float = 1.0
    lambda_cf: float = 0.0
    margin_cf: float = 0.2
    # Per-class weights for the action head CrossEntropyLoss.
    # Order matches ACTION_LABELS: [trust_vision, trust_text, require_agreement, abstain].
    # None = uniform weights (default behaviour).
    action_class_weights: list[float] | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "lambda_vision_info": float(self.lambda_vision_info),
            "lambda_text_info": float(self.lambda_text_info),
            "lambda_relation": float(self.lambda_relation),
            "lambda_action": float(self.lambda_action),
            "lambda_cf": float(self.lambda_cf),
            "margin_cf": float(self.margin_cf),
            "action_class_weights": self.action_class_weights,
            # Backward-compatible aliases for old artifacts/config readers.
            "lambda_info": float((self.lambda_vision_info + self.lambda_text_info) / 2.0),
            "lambda_rel": float(self.lambda_relation),
        }


@dataclass
class ExperimentalTrainerConfig:
    batch_size: int = 4
    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.01
    patience: int = 2
    device: str = "cpu"
    early_stop_metric: str = "task_success_revised"
    log_every_steps: int = 50
    loss: ExperimentalLossConfig = field(default_factory=ExperimentalLossConfig)


@dataclass
class ExperimentalTrainingResult:
    best_model_state_dict: dict[str, torch.Tensor]
    best_val_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    best_epoch: int
    stopped_epoch: int
    early_stop_reason: str
    label_availability: dict[str, int]


class _ExampleDataset(Dataset[ConflictExample]):
    def __init__(self, examples: list[ConflictExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ConflictExample:
        return self.examples[idx]


class ExperimentalTrainer:
    def __init__(
        self,
        *,
        model: ExperimentalCARMHeads | CascadeCARMHeads | DistributionCARMHeads,
        backbone: BackboneAdapter,
        derived_by_id: dict[str, DerivedLabels],
        config: ExperimentalTrainerConfig,
        evaluate_fn: Callable[[ExperimentalCARMHeads, list[ConflictExample], Path], dict[str, Any]],
    ) -> None:
        self.model = model
        self.backbone = backbone
        self.derived_by_id = derived_by_id
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

    def _clear_backbone_caches(self) -> None:
        clear_fn = getattr(self.backbone, "clear_caches", None)
        if callable(clear_fn):
            clear_fn()

    def _forward_example(
        self,
        example: ConflictExample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_payload = example.image_path
        recipe = example.metadata.get("vision_recipe") if isinstance(example.metadata, dict) else None
        if isinstance(recipe, dict) and "payload" in recipe:
            image_payload = str(recipe["payload"])

        with torch.no_grad():
            multimodal = self.backbone.run_backbone_multimodal(image_payload, example.text_input, example.question)
            vision_probe = self.backbone.run_probe_vision_only(image_payload, example.question)
            text_probe = self.backbone.run_probe_text_only(example.text_input, example.question)

        phi_cross = extract_cross_modal_features(
            vision_probe.answer_dist,
            text_probe.answer_dist,
            vision_probe.answer_text,
            text_probe.answer_text,
        ).to(self.device)

        if isinstance(self.model, DistributionCARMHeads):
            vision_info_logits, text_info_logits, relation_logits, action_logits = self.model(
                multimodal.answer_dist.to(self.device),
                vision_probe.answer_dist.to(self.device),
                text_probe.answer_dist.to(self.device),
                phi_cross,
            )
        else:
            vision_info_logits, text_info_logits, relation_logits, action_logits = self.model(
                multimodal.hidden_states.to(self.device),
                vision_probe.features.to(self.device),
                text_probe.features.to(self.device),
                phi_cross,
            )
        return vision_info_logits, text_info_logits, relation_logits, action_logits

    @staticmethod
    def _state_dict_to_cpu(model: ExperimentalCARMHeads) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _label_availability(self, examples: list[ConflictExample]) -> dict[str, int]:
        counts = Counter()
        for example in examples:
            derived = self.derived_by_id[example.example_id]
            counts["examples"] += 1
            counts["vision_info_targets"] += int(derived.vision_info_target_available)
            counts["text_info_targets"] += int(derived.text_info_target_available)
            counts["relation_targets"] += int(derived.relation_target_available)
            counts["action_targets"] += int(derived.action_target_available)
        return dict(counts)

    def _masked_loss(
        self,
        derived: DerivedLabels,
        *,
        vision_info_logits: torch.Tensor,
        text_info_logits: torch.Tensor,
        relation_logits: torch.Tensor,
        action_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, int]]:
        logs = {
            "loss_total": 0.0,
            "loss_vision_info": 0.0,
            "loss_text_info": 0.0,
            "loss_relation": 0.0,
            "loss_action": 0.0,
            "loss_cf": 0.0,
            # Backward-compatible aggregate key.
            "loss_info": 0.0,
        }
        counts = {
            "vision_info_examples": 0,
            "text_info_examples": 0,
            "relation_examples": 0,
            "action_examples": 0,
            "cf_examples": 0,
            # Backward-compatible aggregate key.
            "info_examples": 0,
        }

        total = torch.tensor(0.0, device=self.device)

        if derived.vision_info_target_available and derived.vision_info_state is not None:
            target = torch.tensor([INFO_STATE_TO_IDX[derived.vision_info_state]], device=self.device)
            loss = F.cross_entropy(vision_info_logits, target)
            weighted = float(self.config.loss.lambda_vision_info) * loss
            total = total + weighted
            logs["loss_vision_info"] = float(weighted.item())
            counts["vision_info_examples"] = 1
            counts["info_examples"] += 1

        if derived.text_info_target_available and derived.text_info_state is not None:
            target = torch.tensor([INFO_STATE_TO_IDX[derived.text_info_state]], device=self.device)
            loss = F.cross_entropy(text_info_logits, target)
            weighted = float(self.config.loss.lambda_text_info) * loss
            total = total + weighted
            logs["loss_text_info"] = float(weighted.item())
            counts["text_info_examples"] = 1
            counts["info_examples"] += 1

        logs["loss_info"] = logs["loss_vision_info"] + logs["loss_text_info"]

        if derived.relation_target_available and derived.pairwise_relation is not None:
            target = torch.tensor([PAIRWISE_RELATION_TO_IDX[derived.pairwise_relation]], device=self.device)
            loss = F.cross_entropy(relation_logits, target)
            weighted = float(self.config.loss.lambda_relation) * loss
            total = total + weighted
            logs["loss_relation"] = float(weighted.item())
            counts["relation_examples"] = 1

        if derived.action_target_available and derived.action_target is not None:
            target = torch.tensor([ACTION_TO_IDX[derived.action_target]], device=self.device)
            action_weight = None
            if self.config.loss.action_class_weights is not None:
                action_weight = torch.tensor(self.config.loss.action_class_weights, dtype=torch.float32, device=self.device)
            loss = F.cross_entropy(action_logits, target, weight=action_weight)
            weighted = float(self.config.loss.lambda_action) * loss
            total = total + weighted
            logs["loss_action"] = float(weighted.item())
            counts["action_examples"] = 1

        if float(self.config.loss.lambda_cf) > 0.0:
            raise ValueError("Counterfactual loss is not implemented for the explicit four-head experimental model.")

        logs["loss_total"] = float(total.item())
        return total, logs, counts

    def _train_epoch(
        self,
        epoch: int,
        loader: DataLoader[list[ConflictExample]],
        *,
        progress_file: Any | None = None,
    ) -> dict[str, float]:
        metrics_accum = Counter()
        examples_seen = 0
        steps_done = 0
        total_steps = len(loader)
        log_every_steps = max(0, int(self.config.log_every_steps))
        start_time = time.perf_counter()

        self.model.train()
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.tensor(0.0, device=self.device)

            for example in batch:
                derived = self.derived_by_id[example.example_id]
                vision_info_logits, text_info_logits, relation_logits, action_logits = self._forward_example(example)
                total, logs, counts = self._masked_loss(
                    derived,
                    vision_info_logits=vision_info_logits,
                    text_info_logits=text_info_logits,
                    relation_logits=relation_logits,
                    action_logits=action_logits,
                )
                batch_loss = batch_loss + total
                metrics_accum.update(logs)
                metrics_accum.update(counts)
                examples_seen += 1

            if batch_loss.requires_grad:
                batch_loss.backward()
                self.optimizer.step()
            steps_done += 1

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
                    "avg_loss_total": float(metrics_accum.get("loss_total", 0.0) / max(1, examples_seen)),
                    "avg_loss_vision_info": float(metrics_accum.get("loss_vision_info", 0.0) / max(1, examples_seen)),
                    "avg_loss_text_info": float(metrics_accum.get("loss_text_info", 0.0) / max(1, examples_seen)),
                    "avg_loss_relation": float(metrics_accum.get("loss_relation", 0.0) / max(1, examples_seen)),
                    "avg_loss_action": float(metrics_accum.get("loss_action", 0.0) / max(1, examples_seen)),
                    "avg_loss_cf": float(metrics_accum.get("loss_cf", 0.0) / max(1, examples_seen)),
                }
                line = json.dumps(record, ensure_ascii=True)
                print(line, flush=True)
                if progress_file is not None:
                    progress_file.write(line + "\n")
                    progress_file.flush()

        averaged: dict[str, float | int] = {}
        for key in (
            "loss_total",
            "loss_vision_info",
            "loss_text_info",
            "loss_info",
            "loss_relation",
            "loss_action",
            "loss_cf",
        ):
            averaged[key] = float(metrics_accum.get(key, 0.0) / max(1, examples_seen))
        for key in (
            "vision_info_examples",
            "text_info_examples",
            "info_examples",
            "relation_examples",
            "action_examples",
            "cf_examples",
        ):
            averaged[key] = int(metrics_accum.get(key, 0))
        return averaged

    def _is_improved(self, metrics: dict[str, Any], best_metrics: dict[str, Any] | None) -> bool:
        if best_metrics is None:
            return True
        metric_key = str(self.config.early_stop_metric)
        current = float(metrics.get(metric_key, float("-inf")) or float("-inf"))
        best = float(best_metrics.get(metric_key, float("-inf")) or float("-inf"))
        if current > best:
            return True
        if current < best:
            return False
        return float(metrics.get("action_accuracy", float("-inf")) or float("-inf")) > float(
            best_metrics.get("action_accuracy", float("-inf")) or float("-inf")
        )

    def train(
        self,
        train_examples: list[ConflictExample],
        val_examples: list[ConflictExample],
        *,
        output_dir: str | Path,
    ) -> ExperimentalTrainingResult:
        if not train_examples:
            raise ValueError("Experimental training requires at least one train example.")
        if not val_examples:
            raise ValueError("Experimental training requires at least one validation example.")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(_ExampleDataset(train_examples), batch_size=int(self.config.batch_size), shuffle=True, collate_fn=list)
        label_availability = self._label_availability(train_examples + val_examples)

        history: list[dict[str, Any]] = []
        best_metrics: dict[str, Any] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_epoch = 0
        early_stop_reason = "completed_all_epochs"

        history_path = out_dir / "train_history.jsonl"
        progress_path = out_dir / "train_progress.jsonl"
        best_metrics_path = out_dir / "best_val_metrics.json"
        best_predictions_path = out_dir / "val_predictions_best.jsonl"
        label_availability_path = out_dir / "label_availability.json"
        label_availability_path.write_text(json.dumps(label_availability, indent=2), encoding="utf-8")

        with history_path.open("w", encoding="utf-8") as history_file, progress_path.open("w", encoding="utf-8") as progress_file:
            for epoch in range(1, int(self.config.epochs) + 1):
                train_metrics = self._train_epoch(epoch, loader, progress_file=progress_file)
                self._clear_backbone_caches()

                with tempfile.TemporaryDirectory(dir=out_dir) as temp_dir:
                    val_dir = Path(temp_dir) / "val_eval"
                    val_metrics = self.evaluate_fn(self.model, val_examples, val_dir)
                    self._clear_backbone_caches()

                    improved = self._is_improved(val_metrics, best_metrics)
                    record = {
                        "epoch": epoch,
                        "is_best": bool(improved),
                        **train_metrics,
                        "val_task_success_revised": val_metrics.get("task_success_revised"),
                        "val_action_accuracy": val_metrics.get("action_accuracy"),
                        "val_relation_accuracy": val_metrics.get("relation_accuracy"),
                        "val_vision_info_accuracy": val_metrics.get("vision_info_accuracy"),
                        "val_text_info_accuracy": val_metrics.get("text_info_accuracy"),
                        "val_coverage": val_metrics.get("coverage"),
                        "val_accuracy_on_answered": val_metrics.get("accuracy_on_answered"),
                    }
                    history.append(record)
                    history_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                    history_file.flush()
                    print(json.dumps(record, ensure_ascii=True), flush=True)

                    if improved:
                        best_metrics = dict(val_metrics)
                        best_state = self._state_dict_to_cpu(self.model)
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        predictions_path = val_dir / "per_example_predictions.jsonl"
                        if predictions_path.exists():
                            shutil.copyfile(predictions_path, best_predictions_path)
                    else:
                        epochs_without_improvement += 1

                if best_metrics is not None:
                    best_metrics_payload = {
                        **best_metrics,
                        "best_epoch": best_epoch,
                        "stopped_epoch": epoch,
                        "early_stop_reason": early_stop_reason,
                    }
                    best_metrics_path.write_text(json.dumps(best_metrics_payload, indent=2), encoding="utf-8")

                if epochs_without_improvement >= int(self.config.patience):
                    stopped_epoch = epoch
                    early_stop_reason = (
                        f"no_improvement_for_{int(self.config.patience)}_epochs_on_{self.config.early_stop_metric}"
                    )
                    break
            else:
                stopped_epoch = int(self.config.epochs)

        if best_state is None or best_metrics is None:
            raise RuntimeError("Experimental training completed without a best checkpoint.")

        self.model.load_state_dict(best_state, strict=True)
        best_metrics_payload = {
            **best_metrics,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "early_stop_reason": early_stop_reason,
        }
        best_metrics_path.write_text(json.dumps(best_metrics_payload, indent=2), encoding="utf-8")

        return ExperimentalTrainingResult(
            best_model_state_dict=best_state,
            best_val_metrics=best_metrics_payload,
            history=history,
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            early_stop_reason=early_stop_reason,
            label_availability=label_availability,
        )
