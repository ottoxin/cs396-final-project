from __future__ import annotations

from dataclasses import dataclass

import torch

from carm.data.schema import Action, ConflictExample, Family
from carm.data.vqa_coco import infer_family
from carm.models.interfaces import BackboneAdapter
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

    def __init__(self, backbone: BackboneAdapter) -> None:
        self.backbone = backbone

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        raise NotImplementedError

    @staticmethod
    def _family_from_question(question: str) -> Family:
        return infer_family(question) or Family.EXISTENCE

    @staticmethod
    def _entropy(dist: torch.Tensor) -> float:
        d = torch.clamp(dist, min=1e-9)
        return float((-(d * d.log()).sum()).item())

    @staticmethod
    def _margin(dist: torch.Tensor) -> float:
        vals, _ = torch.topk(dist, k=min(2, dist.numel()))
        if vals.numel() < 2:
            return float(vals[0].item())
        return float(vals[0].item() - vals[1].item())

    @staticmethod
    def _vision_payload(ex: ConflictExample) -> str:
        recipe = ex.metadata.get("vision_recipe") if isinstance(ex.metadata, dict) else None
        if isinstance(recipe, dict) and "payload" in recipe:
            return str(recipe["payload"])
        return ex.image_path


class BackboneDirectBaseline(BaseBaseline):
    name = "backbone_direct"

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        mm = self.backbone.run_backbone_multimodal(self._vision_payload(ex), ex.text_input, ex.question)
        conf = float(torch.max(mm.answer_dist).item())
        return BaselinePrediction(
            pred_conflict_type=Family.NONE.value,
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
        pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        if pv.answer_text != pt.answer_text:
            return BaselinePrediction(
                pred_conflict_type=self._family_from_question(ex.question).value,
                pred_action=Action.ABSTAIN.value,
                final_answer="<ABSTAIN>",
                abstained=True,
                confidence=0.4,
                r_v=0.45,
                r_t=0.45,
            )

        return BaselinePrediction(
            pred_conflict_type=Family.NONE.value,
            pred_action=Action.REQUIRE_AGREEMENT.value,
            final_answer=pv.answer_text,
            abstained=False,
            confidence=0.7,
            r_v=0.7,
            r_t=0.7,
        )


class UncertaintyThresholdAbstainBaseline(BaseBaseline):
    name = "uncertainty_threshold_abstain"

    def __init__(self, backbone: BackboneAdapter, entropy_threshold: float = 1.9) -> None:
        super().__init__(backbone)
        self.entropy_threshold = entropy_threshold

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        mm = self.backbone.run_backbone_multimodal(self._vision_payload(ex), ex.text_input, ex.question)
        ent = self._entropy(mm.answer_dist)
        if ent > self.entropy_threshold:
            return BaselinePrediction(
                pred_conflict_type=self._family_from_question(ex.question).value,
                pred_action=Action.ABSTAIN.value,
                final_answer="<ABSTAIN>",
                abstained=True,
                confidence=max(0.0, 1.0 - ent / 3.0),
                r_v=0.4,
                r_t=0.4,
            )

        return BaselinePrediction(
            pred_conflict_type=Family.NONE.value,
            pred_action=Action.REQUIRE_AGREEMENT.value,
            final_answer=mm.answer_text,
            abstained=False,
            confidence=float(torch.max(mm.answer_dist).item()),
            r_v=0.6,
            r_t=0.6,
        )


class TwoPassSelfConsistencyBaseline(BaseBaseline):
    name = "two_pass_self_consistency"

    def __init__(self, backbone: BackboneAdapter, tie_entropy_delta: float = 0.05) -> None:
        super().__init__(backbone)
        self.tie_entropy_delta = tie_entropy_delta

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        ent_v = self._entropy(pv.answer_dist)
        ent_t = self._entropy(pt.answer_dist)

        if pv.answer_text != pt.answer_text:
            return BaselinePrediction(
                pred_conflict_type=self._family_from_question(ex.question).value,
                pred_action=Action.ABSTAIN.value,
                final_answer="<ABSTAIN>",
                abstained=True,
                confidence=max(0.0, 1.0 - min(ent_v, ent_t) / 3.0),
                r_v=max(0.0, 1.0 - ent_v / 3.0),
                r_t=max(0.0, 1.0 - ent_t / 3.0),
            )

        if abs(ent_v - ent_t) <= self.tie_entropy_delta:
            action = Action.REQUIRE_AGREEMENT
        elif ent_v < ent_t:
            action = Action.TRUST_VISION
        else:
            action = Action.TRUST_TEXT

        final_answer, abstained, _ = apply_action_and_generate(action, pv, pt)
        confidence = max(0.0, 1.0 - min(ent_v, ent_t) / 3.0)

        return BaselinePrediction(
            pred_conflict_type=Family.NONE.value,
            pred_action=action.value,
            final_answer=final_answer,
            abstained=abstained,
            confidence=confidence,
            r_v=max(0.0, 1.0 - ent_v / 3.0),
            r_t=max(0.0, 1.0 - ent_t / 3.0),
        )


class ProbeOnlyHeuristicBaseline(BaseBaseline):
    name = "probe_only_heuristic"

    def __init__(self, backbone: BackboneAdapter, tie_entropy_delta: float = 0.05) -> None:
        super().__init__(backbone)
        self.tie_entropy_delta = tie_entropy_delta

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

        ent_v = self._entropy(pv.answer_dist)
        ent_t = self._entropy(pt.answer_dist)
        margin_v = self._margin(pv.answer_dist)
        margin_t = self._margin(pt.answer_dist)

        if abs(ent_v - ent_t) <= self.tie_entropy_delta:
            if abs(margin_v - margin_t) <= 1e-6:
                action = Action.REQUIRE_AGREEMENT
            elif margin_v > margin_t:
                action = Action.TRUST_VISION
            else:
                action = Action.TRUST_TEXT
        elif ent_v < ent_t:
            action = Action.TRUST_VISION
        else:
            action = Action.TRUST_TEXT

        final_answer, abstained, _ = apply_action_and_generate(action, pv, pt)

        conflict = Family.NONE if pv.answer_text == pt.answer_text else self._family_from_question(ex.question)
        confidence = max(0.0, 1.0 - min(ent_v, ent_t) / 3.0)

        return BaselinePrediction(
            pred_conflict_type=conflict.value,
            pred_action=action.value,
            final_answer=final_answer,
            abstained=abstained,
            confidence=confidence,
            r_v=max(0.0, 1.0 - ent_v / 3.0),
            r_t=max(0.0, 1.0 - ent_t / 3.0),
        )
