from __future__ import annotations

import unittest

import torch

from carm.data.schema import (
    Action,
    AnswerType,
    ConflictExample,
    CorruptModality,
    EvidenceModality,
    Family,
    Operator,
    Split,
)
from carm.eval.baselines import PromptVerificationBaseline
from carm.models.interfaces import ProbeResult


class _PromptTestBackbone:
    name = "prompt_test_backbone"

    def __init__(self, vision_answer: str, text_answer: str) -> None:
        self.vision_answer = vision_answer
        self.text_answer = text_answer

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        return ProbeResult(
            answer_dist=torch.tensor([1.0]),
            answer_text=self.vision_answer,
            features=torch.zeros(3),
        )

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        return ProbeResult(
            answer_dist=torch.tensor([1.0]),
            answer_text=self.text_answer,
            features=torch.zeros(3),
        )

    def run_backbone_multimodal(self, image: str, text: str, question: str):
        raise NotImplementedError


def _example() -> ConflictExample:
    return ConflictExample(
        example_id="e1",
        base_id="b1",
        variant_id="v1",
        image_path="img.jpg",
        text_input="caption",
        question="Is there a bicycle in the image?",
        gold_answer="yes",
        split=Split.VAL,
        family=Family.EXISTENCE,
        operator=Operator.CLEAN,
        corrupt_modality=CorruptModality.NONE,
        severity=0,
        answer_type=AnswerType.BOOLEAN,
        oracle_action=Action.REQUIRE_AGREEMENT,
        evidence_modality=EvidenceModality.EITHER,
    )


class TestPromptVerificationBaseline(unittest.TestCase):
    def test_semantic_equivalent_answers_do_not_abstain(self) -> None:
        baseline = PromptVerificationBaseline(_PromptTestBackbone(vision_answer="yes", text_answer="true"))
        pred = baseline.predict(_example())
        self.assertFalse(pred.abstained)
        self.assertEqual(pred.pred_action, Action.REQUIRE_AGREEMENT.value)


if __name__ == "__main__":
    unittest.main()
