from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from carm.data.labeling import derive_reliability_target
from carm.data.schema import (
    Action,
    ConflictExample,
    ConflictType,
    CorruptedModality,
)


CONFLICT_TO_IDX = {
    ConflictType.NONE: 0,
    ConflictType.OBJECT: 1,
    ConflictType.ATTRIBUTE: 2,
    ConflictType.RELATION: 3,
    ConflictType.COUNT: 4,
}

ACTION_TO_IDX = {
    Action.TRUST_VISION: 0,
    Action.TRUST_TEXT: 1,
    Action.REQUIRE_AGREEMENT: 2,
    Action.ABSTAIN: 3,
}


@dataclass
class TaskLossWeights:
    lambda_cf: float = 0.5
    rel_weight: float = 1.0


@dataclass
class ExampleTargets:
    conflict_idx: int
    action_idx: int
    reliability_target: torch.Tensor


def build_targets(example: ConflictExample, device: torch.device) -> ExampleTargets:
    rt = derive_reliability_target(
        evidence_modality=example.evidence_modality,
        corrupted_modality=example.corrupted_modality,
        severity=example.severity,
    )
    return ExampleTargets(
        conflict_idx=CONFLICT_TO_IDX[example.conflict_type],
        action_idx=ACTION_TO_IDX[example.oracle_action],
        reliability_target=torch.tensor([rt.r_v, rt.r_t], dtype=torch.float32, device=device),
    )


def counterfactual_hinge(
    clean_reliability: torch.Tensor,
    corrupted_reliability: torch.Tensor,
    corrupted_modality: CorruptedModality,
    margin: float = 0.2,
) -> torch.Tensor:
    if corrupted_modality == CorruptedModality.VISION:
        return torch.relu(torch.tensor(margin, device=clean_reliability.device) - (clean_reliability[0] - corrupted_reliability[0]))
    if corrupted_modality == CorruptedModality.TEXT:
        return torch.relu(torch.tensor(margin, device=clean_reliability.device) - (clean_reliability[1] - corrupted_reliability[1]))
    return torch.tensor(0.0, device=clean_reliability.device)


def multi_task_loss(
    conflict_logits: torch.Tensor,
    action_logits: torch.Tensor,
    reliability_pred: torch.Tensor,
    targets: ExampleTargets,
    cf_loss: torch.Tensor,
    weights: TaskLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    conflict = F.cross_entropy(conflict_logits, torch.tensor([targets.conflict_idx], device=conflict_logits.device))
    action = F.cross_entropy(action_logits, torch.tensor([targets.action_idx], device=action_logits.device))
    reliability = F.mse_loss(reliability_pred.squeeze(0), targets.reliability_target)
    total = conflict + action + weights.rel_weight * reliability + weights.lambda_cf * cf_loss
    return total, {
        "loss_total": float(total.item()),
        "loss_conflict": float(conflict.item()),
        "loss_action": float(action.item()),
        "loss_reliability": float(reliability.item()),
        "loss_cf": float(cf_loss.item()),
    }
