from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from carm.models.features import extract_probe_features
from carm.models.interfaces import BackboneResult, FreeformGenerationResult, ProbeResult


@dataclass
class DebugBackboneConfig:
    hidden_size: int = 128
    seq_len: int = 32
    vocab: tuple[str, ...] = (
        "yes",
        "no",
        "red",
        "blue",
        "green",
        "0",
        "1",
        "2",
        "3",
        "unknown",
    )
    action_vocab: tuple[str, ...] = (
        "TRUST_VISION",
        "TRUST_TEXT",
        "REQUIRE_AGREEMENT",
        "ABSTAIN",
    )


class DeterministicDebugBackbone:
    """Deterministic CPU-safe adapter for small-run preflights."""

    name = "deterministic_debug_backbone"

    def __init__(self, config: DebugBackboneConfig | None = None) -> None:
        self.config = config or DebugBackboneConfig()
        self.device = torch.device("cpu")

    @staticmethod
    def _seed_from_payload(payload: str) -> int:
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)

    def _sample_distribution(self, payload: str, vocab_size: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed_from_payload(payload))
        logits = torch.randn(vocab_size, generator=generator)
        return torch.softmax(logits, dim=-1)

    def _hidden_states(self, payload: str) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed_from_payload(f"hidden::{payload}"))
        return torch.randn(self.config.seq_len, self.config.hidden_size, generator=generator)

    @staticmethod
    def _confidence(dist: torch.Tensor) -> float:
        return float(torch.max(dist).item()) if dist.numel() else 0.0

    def _decode(self, dist: torch.Tensor) -> str:
        return self.config.vocab[int(torch.argmax(dist).item())]

    @staticmethod
    def _metadata(answer: str) -> dict[str, object]:
        return {
            "projection_succeeded": True,
            "used_fallback_dist": False,
            "parsed_unknown": answer == "unknown",
            "parsed_in_active_vocab": answer != "unknown",
            "canonicalized_candidate": None if answer == "unknown" else answer,
            "out_of_vocab_generation": False,
            "dist_argmax_label": answer,
            "parsed_argmax_agree": True,
        }

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        payload = f"mm::{image}::{text}::{question}"
        dist = self._sample_distribution(payload, len(self.config.vocab))
        answer = self._decode(dist)
        return BackboneResult(
            hidden_states=self._hidden_states(payload),
            answer_dist=dist,
            answer_text=answer,
            raw_text=f"debug::{answer}",
            metadata=self._metadata(answer),
        )

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        payload = f"vision::{image}::{question}"
        dist = self._sample_distribution(payload, len(self.config.vocab))
        answer = self._decode(dist)
        return ProbeResult(
            answer_dist=dist,
            answer_text=answer,
            features=extract_probe_features(dist),
            raw_text=f"debug::{answer}",
            metadata=self._metadata(answer),
        )

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        payload = f"text::{text}::{question}"
        dist = self._sample_distribution(payload, len(self.config.vocab))
        answer = self._decode(dist)
        return ProbeResult(
            answer_dist=dist,
            answer_text=answer,
            features=extract_probe_features(dist),
            raw_text=f"debug::{answer}",
            metadata=self._metadata(answer),
        )

    def generate_freeform(self, prompt: str, image: str | None = None) -> FreeformGenerationResult:
        payload = f"freeform::{image or '<none>'}::{prompt}"
        dist = self._sample_distribution(payload, len(self.config.action_vocab))
        token = self.config.action_vocab[int(torch.argmax(dist).item())]
        return FreeformGenerationResult(
            text=token,
            confidence=self._confidence(dist),
            metadata={
                "generator": self.name,
                "action_vocab": list(self.config.action_vocab),
            },
        )
