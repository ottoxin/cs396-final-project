from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from carm.data.schema import Action


@dataclass
class BackboneResult:
    hidden_states: torch.Tensor  # [L, D]
    answer_dist: torch.Tensor  # [V]
    answer_text: str


@dataclass
class ProbeResult:
    answer_dist: torch.Tensor  # [V]
    answer_text: str
    features: torch.Tensor  # [F]


class BackboneAdapter(Protocol):
    name: str

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        ...

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        ...

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        ...


@dataclass
class CARMOutput:
    conflict_logits: torch.Tensor
    reliability: dict[str, float]
    action_logits: torch.Tensor
    action: Action
    final_answer: str
    abstained: bool
