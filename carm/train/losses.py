from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from carm.data.labeling import derive_reliability_target
from carm.data.schema import Action, ConflictExample, CorruptModality, Family


ACTION_LABELS = (
    Action.TRUST_VISION.value,
    Action.TRUST_TEXT.value,
    Action.REQUIRE_AGREEMENT.value,
    Action.ABSTAIN.value,
)

CONFLICT_TO_IDX = {
    Family.NONE: 0,
    Family.EXISTENCE: 1,
    Family.COUNT: 2,
    Family.ATTRIBUTE_COLOR: 3,
}

ACTION_TO_IDX = {
    Action.TRUST_VISION: 0,
    Action.TRUST_TEXT: 1,
    Action.REQUIRE_AGREEMENT: 2,
    Action.ABSTAIN: 3,
}


@dataclass
class LossConfig:
    action: bool = True
    conflict: bool = False
    reliability: bool = False
    counterfactual: bool = False
    lambda_conf: float = 0.0
    lambda_rel: float = 0.0
    lambda_cf: float = 0.0
    margin_cf: float = 0.2

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        legacy_training: Mapping[str, Any] | None = None,
    ) -> LossConfig:
        if raw is not None:
            action = bool(raw.get("action", True))
            conflict = bool(raw.get("conflict", False))
            reliability = bool(raw.get("reliability", False))
            counterfactual = bool(raw.get("counterfactual", False))
            lambda_conf = float(raw.get("lambda_conf", 1.0 if conflict else 0.0))
            lambda_rel = float(raw.get("lambda_rel", 1.0 if reliability else 0.0))
            lambda_cf = float(raw.get("lambda_cf", 0.5 if counterfactual else 0.0))
            margin_cf = float(raw.get("margin_cf", 0.2))
            return cls(
                action=action,
                conflict=conflict,
                reliability=reliability,
                counterfactual=counterfactual,
                lambda_conf=lambda_conf,
                lambda_rel=lambda_rel,
                lambda_cf=lambda_cf,
                margin_cf=margin_cf,
            )

        legacy = legacy_training or {}
        return cls(
            action=True,
            conflict=True,
            reliability=True,
            counterfactual=float(legacy.get("lambda_cf", 0.5)) > 0.0,
            lambda_conf=1.0,
            lambda_rel=1.0,
            lambda_cf=float(legacy.get("lambda_cf", 0.5)),
            margin_cf=float(legacy.get("margin_cf", 0.2)),
        )

    def __post_init__(self) -> None:
        self.action = bool(self.action)
        self.conflict = bool(self.conflict)
        self.reliability = bool(self.reliability)
        self.counterfactual = bool(self.counterfactual)
        self.lambda_conf = float(self.lambda_conf)
        self.lambda_rel = float(self.lambda_rel)
        self.lambda_cf = float(self.lambda_cf)
        self.margin_cf = float(self.margin_cf)

        if not (self.action or self.conflict or self.reliability or self.counterfactual):
            raise ValueError("At least one loss must be enabled.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def enabled_losses(self) -> dict[str, bool]:
        return {
            "action": self.action,
            "conflict": self.conflict,
            "reliability": self.reliability,
            "counterfactual": self.counterfactual,
        }

    def diagnostic_validity(self) -> dict[str, bool]:
        return {
            "conflict": bool(self.conflict),
            "reliability": bool(self.reliability or self.counterfactual),
        }


@dataclass
class ExampleTargets:
    conflict_idx: int
    action_idx: int
    reliability_target: torch.Tensor


def build_targets(example: ConflictExample, device: torch.device) -> ExampleTargets:
    rt = derive_reliability_target(
        evidence_modality=example.evidence_modality,
        corrupt_modality=example.corrupt_modality,
        severity=example.severity,
    )
    return ExampleTargets(
        conflict_idx=CONFLICT_TO_IDX[example.family],
        action_idx=ACTION_TO_IDX[example.oracle_action],
        reliability_target=torch.tensor([rt.r_v, rt.r_t], dtype=torch.float32, device=device),
    )


def counterfactual_hinge(
    clean_reliability: torch.Tensor,
    corrupted_reliability: torch.Tensor,
    corrupted_modality: CorruptModality,
    margin: float = 0.2,
) -> torch.Tensor:
    if corrupted_modality == CorruptModality.VISION:
        return torch.relu(
            torch.tensor(margin, device=clean_reliability.device)
            - (clean_reliability[0] - corrupted_reliability[0])
        )
    if corrupted_modality == CorruptModality.TEXT:
        return torch.relu(
            torch.tensor(margin, device=clean_reliability.device)
            - (clean_reliability[1] - corrupted_reliability[1])
        )
    if corrupted_modality == CorruptModality.BOTH:
        loss_v = torch.relu(
            torch.tensor(margin, device=clean_reliability.device)
            - (clean_reliability[0] - corrupted_reliability[0])
        )
        loss_t = torch.relu(
            torch.tensor(margin, device=clean_reliability.device)
            - (clean_reliability[1] - corrupted_reliability[1])
        )
        return 0.5 * (loss_v + loss_t)
    return torch.tensor(0.0, device=clean_reliability.device)


def multi_task_loss(
    conflict_logits: torch.Tensor,
    action_logits: torch.Tensor,
    reliability_pred: torch.Tensor,
    targets: ExampleTargets,
    cf_loss: torch.Tensor,
    loss_cfg: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    conflict = F.cross_entropy(
        conflict_logits,
        torch.tensor([targets.conflict_idx], device=conflict_logits.device),
    )
    action = F.cross_entropy(
        action_logits,
        torch.tensor([targets.action_idx], device=action_logits.device),
    )
    reliability = F.mse_loss(reliability_pred.squeeze(0), targets.reliability_target)
    total = torch.tensor(0.0, device=action_logits.device)
    if loss_cfg.action:
        total = total + action
    if loss_cfg.conflict:
        total = total + (loss_cfg.lambda_conf * conflict)
    if loss_cfg.reliability:
        total = total + (loss_cfg.lambda_rel * reliability)
    if loss_cfg.counterfactual:
        total = total + (loss_cfg.lambda_cf * cf_loss)
    return total, {
        "loss_total": float(total.item()),
        "loss_conflict": float((loss_cfg.lambda_conf * conflict).item()) if loss_cfg.conflict else 0.0,
        "loss_action": float(action.item()) if loss_cfg.action else 0.0,
        "loss_reliability": float((loss_cfg.lambda_rel * reliability).item()) if loss_cfg.reliability else 0.0,
        "loss_cf": float((loss_cfg.lambda_cf * cf_loss).item()) if loss_cfg.counterfactual else 0.0,
    }
