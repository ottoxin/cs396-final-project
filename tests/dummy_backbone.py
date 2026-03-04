from __future__ import annotations

import hashlib

import torch

from carm.models.backbone import BackboneConfig
from carm.models.features import extract_probe_features
from carm.models.interfaces import BackboneResult, ProbeResult


class DeterministicTestBackbone:
    """Lightweight deterministic adapter used only in unit tests."""

    name = "deterministic_test_backbone"

    def __init__(self, config: BackboneConfig | None = None) -> None:
        self.config = config or BackboneConfig()

    @staticmethod
    def _seed_from_payload(payload: str) -> int:
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)

    def _sample_distribution(self, payload: str) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self._seed_from_payload(payload))
        logits = torch.randn(len(self.config.vocab), generator=g)
        return torch.softmax(logits, dim=-1)

    def _hidden_states(self, payload: str) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self._seed_from_payload("hs::" + payload))
        return torch.randn(self.config.seq_len, self.config.hidden_size, generator=g)

    def _decode(self, dist: torch.Tensor) -> str:
        return self.config.vocab[int(torch.argmax(dist).item())]

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        payload = f"mm::{image}::{text}::{question}"
        dist = self._sample_distribution(payload)
        return BackboneResult(
            hidden_states=self._hidden_states(payload),
            answer_dist=dist,
            answer_text=self._decode(dist),
        )

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        payload = f"v::{image}::{question}"
        dist = self._sample_distribution(payload)
        return ProbeResult(
            answer_dist=dist,
            answer_text=self._decode(dist),
            features=extract_probe_features(dist),
        )

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        payload = f"t::{text}::{question}"
        dist = self._sample_distribution(payload)
        return ProbeResult(
            answer_dist=dist,
            answer_text=self._decode(dist),
            features=extract_probe_features(dist),
        )
