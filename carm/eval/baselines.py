from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from carm.data.schema import ConflictExample
from carm.models.interfaces import BackboneAdapter
from carm.models.policy import answers_agree, normalize_answer


ABSTAIN_ANSWER = "<ABSTAIN>"


@dataclass
class BaselinePrediction:
    final_answer: str
    abstained: bool
    confidence: float


class BaseBaseline:
    name = "base"

    def __init__(self, backbone: BackboneAdapter) -> None:
        self.backbone = backbone

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        raise NotImplementedError

    @staticmethod
    def _vision_payload(ex: ConflictExample) -> str:
        recipe = ex.metadata.get("vision_recipe") if isinstance(ex.metadata, dict) else None
        if isinstance(recipe, dict) and "payload" in recipe:
            return str(recipe["payload"])
        return ex.image_path

    @staticmethod
    def _entropy(dist: torch.Tensor) -> float:
        probs = torch.clamp(dist, min=1e-12)
        return float((-(probs * probs.log()).sum()).item())

    @classmethod
    def _normalized_entropy(cls, dist: torch.Tensor) -> float:
        if dist.numel() <= 1:
            return 0.0
        max_entropy = math.log(float(dist.numel()))
        if max_entropy <= 0.0:
            return 0.0
        return max(0.0, min(1.0, cls._entropy(dist) / max_entropy))

    @staticmethod
    def _max_prob(dist: torch.Tensor) -> float:
        return float(torch.max(dist).item())

    @staticmethod
    def _abstain(confidence: float) -> BaselinePrediction:
        return BaselinePrediction(
            final_answer=ABSTAIN_ANSWER,
            abstained=True,
            confidence=max(0.0, min(1.0, float(confidence))),
        )


class BackboneDirectBaseline(BaseBaseline):
    name = "backbone_direct"

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        mm = self.backbone.run_backbone_multimodal(self._vision_payload(ex), ex.text_input, ex.question)
        return BaselinePrediction(
            final_answer=mm.answer_text,
            abstained=False,
            confidence=self._max_prob(mm.answer_dist),
        )


class AgreementCheckBaseline(BaseBaseline):
    name = "agreement_check"

    def _disagreement_confidence(self, vision_dist: torch.Tensor, text_dist: torch.Tensor) -> float:
        norm_v = self._normalized_entropy(vision_dist)
        norm_t = self._normalized_entropy(text_dist)
        return 0.5 * max(norm_v, norm_t)

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        if answers_agree(pv.answer_text, pt.answer_text):
            return BaselinePrediction(
                final_answer=normalize_answer(pv.answer_text),
                abstained=False,
                confidence=min(self._max_prob(pv.answer_dist), self._max_prob(pt.answer_dist)),
            )

        return self._abstain(self._disagreement_confidence(pv.answer_dist, pt.answer_dist))


class ConfidenceThresholdBaseline(BaseBaseline):
    name = "confidence_threshold"

    def __init__(self, backbone: BackboneAdapter, threshold: float = 0.3) -> None:
        super().__init__(backbone)
        self.threshold = float(threshold)

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        mm = self.backbone.run_backbone_multimodal(self._vision_payload(ex), ex.text_input, ex.question)
        confidence = 1.0 - self._normalized_entropy(mm.answer_dist)
        if confidence < self.threshold:
            return self._abstain(confidence)
        return BaselinePrediction(
            final_answer=mm.answer_text,
            abstained=False,
            confidence=confidence,
        )


class ProbeHeuristicBaseline(BaseBaseline):
    name = "probe_heuristic"

    def __init__(self, backbone: BackboneAdapter, both_uncertain_threshold: float = 2.0) -> None:
        super().__init__(backbone)
        self.both_uncertain_threshold = float(both_uncertain_threshold)

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        ent_v = self._entropy(pv.answer_dist)
        ent_t = self._entropy(pt.answer_dist)
        chosen = pv if ent_v <= ent_t else pt
        confidence = self._max_prob(chosen.answer_dist)

        if ent_v > self.both_uncertain_threshold and ent_t > self.both_uncertain_threshold:
            return self._abstain(confidence)

        return BaselinePrediction(
            final_answer=chosen.answer_text,
            abstained=False,
            confidence=confidence,
        )
