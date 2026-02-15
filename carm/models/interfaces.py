from __future__ import annotations

from dataclasses import dataclass

import torch

from carm.data.schema import Action


@dataclass
class BackboneResult:
    hidden_states: torch.Tensor  # [L, D]
    answer_dist: torch.Tensor  # [V]
    answer_text: str


@dataclass
class ProbeResult:
    answer_dist: torch.Tensor
    answer_text: str
    features: torch.Tensor  # [F]


@dataclass
class CARMOutput:
    conflict_logits: torch.Tensor  # [5]
    reliability: dict[str, float]
    action_logits: torch.Tensor  # [4]
    action: Action
    final_answer: str
    abstained: bool
