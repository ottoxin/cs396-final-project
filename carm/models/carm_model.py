from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from carm.data.schema import Action


@dataclass
class CARMModelConfig:
    hidden_size: int = 128
    probe_feature_size: int = 3
    pool: str = "mean"


class CARMHeads(nn.Module):
    def __init__(self, config: CARMModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or CARMModelConfig()
        decision_dim = self.config.hidden_size + (2 * self.config.probe_feature_size)

        self.conflict_head = nn.Linear(decision_dim, 4)
        self.reliability_head = nn.Linear(decision_dim, 2)
        self.action_head = nn.Linear(decision_dim, 4)

    def pool_anchor_states(self, anchor_states: torch.Tensor) -> torch.Tensor:
        if anchor_states.dim() == 3:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=1)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        if anchor_states.dim() == 2:
            if self.config.pool == "mean":
                return anchor_states.mean(dim=0)
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")
        raise ValueError("anchor_states must have 2 or 3 dims")

    def carm_forward(
        self,
        anchor_states: torch.Tensor,
        phi_v: torch.Tensor,
        phi_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pool_anchor_states(anchor_states)
        if h.dim() == 1:
            h = h.unsqueeze(0)

        if phi_v.dim() == 1:
            phi_v = phi_v.unsqueeze(0)
        if phi_t.dim() == 1:
            phi_t = phi_t.unsqueeze(0)

        u = torch.cat([h, phi_v, phi_t], dim=-1)
        conflict_logits = self.conflict_head(u)
        reliability = torch.sigmoid(self.reliability_head(u))
        action_logits = self.action_head(u)
        return conflict_logits, reliability, action_logits


def select_action(action_logits: torch.Tensor) -> Action:
    idx = int(torch.argmax(action_logits, dim=-1).item())
    mapping = [
        Action.TRUST_VISION,
        Action.TRUST_TEXT,
        Action.REQUIRE_AGREEMENT,
        Action.ABSTAIN,
    ]
    return mapping[idx]
