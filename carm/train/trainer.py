from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from carm.data.io import save_examples
from carm.data.schema import ConflictExample, CorruptedModality
from carm.models.backbone import MockFrozenBackbone
from carm.models.carm_model import CARMHeads
from carm.train.dataset import ConflictDataset, build_clean_index, pair_key
from carm.train.losses import (
    TaskLossWeights,
    build_targets,
    counterfactual_hinge,
    multi_task_loss,
)


@dataclass
class TrainerConfig:
    batch_size: int = 4
    epochs: int = 2
    lr: float = 1e-3
    lambda_cf: float = 0.5
    margin_cf: float = 0.2
    device: str = "cpu"


class CARMTrainer:
    def __init__(
        self,
        model: CARMHeads,
        backbone: MockFrozenBackbone,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.backbone = backbone
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _forward_example(self, ex: ConflictExample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mm = self.backbone.run_backbone_multimodal(ex.image_path, ex.text_input, ex.question)
        pv = self.backbone.run_probe_vision_only(ex.image_path, ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        anchor_states = mm.hidden_states.to(self.device)
        phi_v = pv.features.to(self.device)
        phi_t = pt.features.to(self.device)
        conflict_logits, reliability, action_logits = self.model.carm_forward(anchor_states, phi_v, phi_t)
        return conflict_logits, reliability, action_logits

    def train(
        self,
        examples: list[ConflictExample],
        output_dir: str | Path,
    ) -> dict[str, float]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = ConflictDataset(examples)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=list)
        clean_index = build_clean_index(examples)

        metrics_accum: dict[str, float] = {
            "loss_total": 0.0,
            "loss_conflict": 0.0,
            "loss_action": 0.0,
            "loss_reliability": 0.0,
            "loss_cf": 0.0,
        }
        steps = 0

        weights = TaskLossWeights(lambda_cf=self.config.lambda_cf)

        self.model.train()
        for _ in range(self.config.epochs):
            for batch in loader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss = torch.tensor(0.0, device=self.device)
                batch_logs: list[dict[str, float]] = []

                for ex in batch:
                    conflict_logits, reliability, action_logits = self._forward_example(ex)
                    targets = build_targets(ex, device=self.device)

                    cf = torch.tensor(0.0, device=self.device)
                    if ex.corrupted_modality != CorruptedModality.NONE:
                        ref = clean_index.get(pair_key(ex))
                        if ref is not None:
                            _, clean_rel, _ = self._forward_example(ref)
                            cf = counterfactual_hinge(
                                clean_reliability=clean_rel.squeeze(0),
                                corrupted_reliability=reliability.squeeze(0),
                                corrupted_modality=ex.corrupted_modality,
                                margin=self.config.margin_cf,
                            )

                    total, logs = multi_task_loss(
                        conflict_logits=conflict_logits,
                        action_logits=action_logits,
                        reliability_pred=reliability,
                        targets=targets,
                        cf_loss=cf,
                        weights=weights,
                    )
                    batch_loss = batch_loss + total
                    batch_logs.append(logs)

                batch_loss.backward()
                self.optimizer.step()

                steps += 1
                for logs in batch_logs:
                    for key, value in logs.items():
                        metrics_accum[key] += value

        averaged = {k: (v / max(1, steps)) for k, v in metrics_accum.items()}

        with (out_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(averaged, f, indent=2)

        # Snapshot effective training set for reproducibility.
        save_examples(out_dir / "train_examples_snapshot.jsonl", examples)
        return averaged
