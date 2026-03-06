from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from carm.data.schema import ConflictExample
from carm.models.interfaces import BackboneAdapter
from carm.models.policy import answers_agree, normalize_answer


ABSTAIN_ANSWER = "<ABSTAIN>"
DIAGNOSTIC_BOOL_KEYS = (
    "projection_succeeded",
    "used_fallback_dist",
    "parsed_unknown",
    "parsed_in_active_vocab",
    "out_of_vocab_generation",
    "parsed_argmax_agree",
)
DIAGNOSTIC_VALUE_KEYS = (
    "canonicalized_candidate",
    "dist_argmax_label",
)


@dataclass
class BaselinePrediction:
    final_answer: str
    abstained: bool
    confidence: float
    raw_text: str | None = None
    metadata: dict[str, Any] | None = None


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
    def _result_metadata(result: Any) -> dict[str, Any]:
        return dict(getattr(result, "metadata", None) or {})

    @classmethod
    def _merge_probe_metadata(cls, vision_result: Any, text_result: Any) -> dict[str, Any]:
        vision_meta = cls._result_metadata(vision_result)
        text_meta = cls._result_metadata(text_result)
        merged: dict[str, Any] = {
            "vision_raw_output": getattr(vision_result, "raw_text", None),
            "text_raw_output": getattr(text_result, "raw_text", None),
        }
        for key in DIAGNOSTIC_BOOL_KEYS:
            v_val = bool(vision_meta.get(key, False))
            t_val = bool(text_meta.get(key, False))
            if key in {"projection_succeeded", "parsed_in_active_vocab", "parsed_argmax_agree"}:
                merged[key] = v_val and t_val
            else:
                merged[key] = v_val or t_val
        for key in DIAGNOSTIC_VALUE_KEYS:
            v_val = vision_meta.get(key)
            t_val = text_meta.get(key)
            merged[key] = v_val if v_val == t_val else None
        return merged

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
        metadata = self._result_metadata(mm)
        metadata["multimodal_raw_output"] = mm.raw_text
        return BaselinePrediction(
            final_answer=mm.answer_text,
            abstained=False,
            confidence=self._max_prob(mm.answer_dist),
            raw_text=mm.raw_text,
            metadata=metadata,
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
        merged_meta = self._merge_probe_metadata(pv, pt)

        if answers_agree(pv.answer_text, pt.answer_text):
            return BaselinePrediction(
                final_answer=normalize_answer(pv.answer_text),
                abstained=False,
                confidence=min(self._max_prob(pv.answer_dist), self._max_prob(pt.answer_dist)),
                raw_text=pv.raw_text,
                metadata=merged_meta,
            )

        pred = self._abstain(self._disagreement_confidence(pv.answer_dist, pt.answer_dist))
        pred.metadata = merged_meta
        return pred


class ConfidenceThresholdBaseline(BaseBaseline):
    name = "confidence_threshold"

    def __init__(self, backbone: BackboneAdapter, threshold: float = 0.3) -> None:
        super().__init__(backbone)
        self.threshold = float(threshold)

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        mm = self.backbone.run_backbone_multimodal(self._vision_payload(ex), ex.text_input, ex.question)
        metadata = self._result_metadata(mm)
        metadata["multimodal_raw_output"] = mm.raw_text
        confidence = 1.0 - self._normalized_entropy(mm.answer_dist)
        if confidence < self.threshold:
            pred = self._abstain(confidence)
            pred.raw_text = mm.raw_text
            pred.metadata = metadata
            return pred
        return BaselinePrediction(
            final_answer=mm.answer_text,
            abstained=False,
            confidence=confidence,
            raw_text=mm.raw_text,
            metadata=metadata,
        )


class ProbeHeuristicBaseline(BaseBaseline):
    name = "probe_heuristic"

    def __init__(self, backbone: BackboneAdapter, both_uncertain_threshold: float = 2.0) -> None:
        super().__init__(backbone)
        self.both_uncertain_threshold = float(both_uncertain_threshold)

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)
        chosen_meta = self._result_metadata(pv if self._entropy(pv.answer_dist) <= self._entropy(pt.answer_dist) else pt)

        ent_v = self._entropy(pv.answer_dist)
        ent_t = self._entropy(pt.answer_dist)
        chosen = pv if ent_v <= ent_t else pt
        confidence = self._max_prob(chosen.answer_dist)
        chosen_meta["vision_raw_output"] = pv.raw_text
        chosen_meta["text_raw_output"] = pt.raw_text

        if ent_v > self.both_uncertain_threshold and ent_t > self.both_uncertain_threshold:
            pred = self._abstain(confidence)
            pred.raw_text = chosen.raw_text
            pred.metadata = chosen_meta
            return pred

        return BaselinePrediction(
            final_answer=chosen.answer_text,
            abstained=False,
            confidence=confidence,
            raw_text=chosen.raw_text,
            metadata=chosen_meta,
        )
