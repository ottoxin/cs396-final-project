from __future__ import annotations

from dataclasses import dataclass

import torch

from carm.data.schema import Action, ConflictExample, ConflictType
from carm.models.backbone import MockFrozenBackbone
from carm.models.policy import apply_action_and_generate


@dataclass
class BaselinePrediction:
    pred_conflict_type: str
    pred_action: str
    final_answer: str
    abstained: bool
    confidence: float
    r_v: float
    r_t: float


class BaseBaseline:
    name = "base"

    def __init__(self, backbone: MockFrozenBackbone) -> None:
        self.backbone = backbone

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        raise NotImplementedError


class BackboneOnlyBaseline(BaseBaseline):
    name = "backbone_only"

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        mm = self.backbone.run_backbone_multimodal(ex.image_path, ex.text_input, ex.question)
        conf = float(torch.max(mm.answer_dist).item())
        return BaselinePrediction(
            pred_conflict_type=ConflictType.NONE.value,
            pred_action=Action.REQUIRE_AGREEMENT.value,
            final_answer=mm.answer_text,
            abstained=False,
            confidence=conf,
            r_v=0.5,
            r_t=0.5,
        )


class PromptVerificationBaseline(BaseBaseline):
    name = "prompt_verification"

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        v = self.backbone.run_probe_vision_only(ex.image_path, ex.question)
        t = self.backbone.run_probe_text_only(ex.text_input, ex.question)
        if v.answer_text != t.answer_text:
            return BaselinePrediction(
                pred_conflict_type=ConflictType.OBJECT.value,
                pred_action=Action.ABSTAIN.value,
                final_answer="I cannot verify consistency between modalities.",
                abstained=True,
                confidence=0.4,
                r_v=0.45,
                r_t=0.45,
            )

        return BaselinePrediction(
            pred_conflict_type=ConflictType.NONE.value,
            pred_action=Action.REQUIRE_AGREEMENT.value,
            final_answer=v.answer_text,
            abstained=False,
            confidence=0.7,
            r_v=0.7,
            r_t=0.7,
        )


class UncertaintyThresholdAbstainBaseline(BaseBaseline):
    name = "uncertainty_threshold_abstain"

    def __init__(self, backbone: MockFrozenBackbone, entropy_threshold: float = 1.9) -> None:
        super().__init__(backbone)
        self.entropy_threshold = entropy_threshold

    def _entropy(self, dist: torch.Tensor) -> float:
        d = torch.clamp(dist, min=1e-9)
        return float((-(d * d.log()).sum()).item())

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        mm = self.backbone.run_backbone_multimodal(ex.image_path, ex.text_input, ex.question)
        ent = self._entropy(mm.answer_dist)
        if ent > self.entropy_threshold:
            return BaselinePrediction(
                pred_conflict_type=ConflictType.OBJECT.value,
                pred_action=Action.ABSTAIN.value,
                final_answer="I am uncertain and will abstain.",
                abstained=True,
                confidence=max(0.0, 1.0 - ent / 3.0),
                r_v=0.4,
                r_t=0.4,
            )

        return BaselinePrediction(
            pred_conflict_type=ConflictType.NONE.value,
            pred_action=Action.REQUIRE_AGREEMENT.value,
            final_answer=mm.answer_text,
            abstained=False,
            confidence=float(torch.max(mm.answer_dist).item()),
            r_v=0.6,
            r_t=0.6,
        )


class ProbeOnlyHeuristicBaseline(BaseBaseline):
    name = "probe_only_heuristic"

    def _entropy(self, dist: torch.Tensor) -> float:
        d = torch.clamp(dist, min=1e-9)
        return float((-(d * d.log()).sum()).item())

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        pv = self.backbone.run_probe_vision_only(ex.image_path, ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        ent_v = self._entropy(pv.answer_dist)
        ent_t = self._entropy(pt.answer_dist)

        if abs(ent_v - ent_t) < 0.05:
            action = Action.REQUIRE_AGREEMENT
        elif ent_v < ent_t:
            action = Action.TRUST_VISION
        else:
            action = Action.TRUST_TEXT

        final_answer, abstained, _ = apply_action_and_generate(action, pv, pt)

        conflict = ConflictType.NONE if pv.answer_text == pt.answer_text else ConflictType.OBJECT
        conf = max(0.0, 1.0 - min(ent_v, ent_t) / 3.0)

        return BaselinePrediction(
            pred_conflict_type=conflict.value,
            pred_action=action.value,
            final_answer=final_answer,
            abstained=abstained,
            confidence=conf,
            r_v=max(0.0, 1.0 - ent_v / 3.0),
            r_t=max(0.0, 1.0 - ent_t / 3.0),
        )
