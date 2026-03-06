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
from carm.eval.baselines import (
    AgreementCheckBaseline,
    BackboneDirectBaseline,
    ConfidenceThresholdBaseline,
    ProbeHeuristicBaseline,
)
from carm.models.interfaces import BackboneResult, ProbeResult


class _ControlledBackbone:
    name = "controlled_backbone"

    def __init__(
        self,
        *,
        mm_answer: str = "yes",
        mm_raw: str | None = None,
        mm_dist: torch.Tensor | None = None,
        vision_answer: str = "yes",
        vision_raw: str | None = None,
        vision_dist: torch.Tensor | None = None,
        text_answer: str = "yes",
        text_raw: str | None = None,
        text_dist: torch.Tensor | None = None,
    ) -> None:
        self.mm_answer = mm_answer
        self.mm_raw = mm_raw if mm_raw is not None else f"raw::{mm_answer}"
        self.mm_dist = mm_dist if mm_dist is not None else torch.tensor([0.9, 0.1], dtype=torch.float32)
        self.vision_answer = vision_answer
        self.vision_raw = vision_raw if vision_raw is not None else f"raw::{vision_answer}"
        self.vision_dist = vision_dist if vision_dist is not None else torch.tensor([0.9, 0.1], dtype=torch.float32)
        self.text_answer = text_answer
        self.text_raw = text_raw if text_raw is not None else f"raw::{text_answer}"
        self.text_dist = text_dist if text_dist is not None else torch.tensor([0.9, 0.1], dtype=torch.float32)

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        return BackboneResult(
            hidden_states=torch.zeros(2, 3),
            answer_dist=self.mm_dist,
            answer_text=self.mm_answer,
            raw_text=self.mm_raw,
        )

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        return ProbeResult(
            answer_dist=self.vision_dist,
            answer_text=self.vision_answer,
            features=torch.zeros(3),
            raw_text=self.vision_raw,
        )

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        return ProbeResult(
            answer_dist=self.text_dist,
            answer_text=self.text_answer,
            features=torch.zeros(3),
            raw_text=self.text_raw,
        )


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
        metadata={"protocol_category": "C1"},
    )


class TestBaselines(unittest.TestCase):
    def test_backbone_direct_returns_multimodal_answer(self) -> None:
        baseline = BackboneDirectBaseline(
            _ControlledBackbone(mm_answer="blue", mm_dist=torch.tensor([0.2, 0.8], dtype=torch.float32))
        )

        pred = baseline.predict(_example())

        self.assertEqual(pred.final_answer, "blue")
        self.assertFalse(pred.abstained)
        self.assertAlmostEqual(pred.confidence, 0.8, places=6)
        self.assertEqual(pred.raw_text, "raw::blue")

    def test_agreement_check_returns_agreed_answer(self) -> None:
        baseline = AgreementCheckBaseline(
            _ControlledBackbone(
                vision_answer="yes",
                vision_dist=torch.tensor([0.8, 0.2], dtype=torch.float32),
                text_answer="true",
                text_dist=torch.tensor([0.7, 0.3], dtype=torch.float32),
            )
        )

        pred = baseline.predict(_example())

        self.assertEqual(pred.final_answer, "yes")
        self.assertFalse(pred.abstained)
        self.assertAlmostEqual(pred.confidence, 0.7, places=6)
        self.assertEqual(pred.raw_text, "raw::yes")
        self.assertEqual(pred.metadata["vision_raw_output"], "raw::yes")
        self.assertEqual(pred.metadata["text_raw_output"], "raw::true")

    def test_agreement_check_abstains_on_disagreement(self) -> None:
        baseline = AgreementCheckBaseline(
            _ControlledBackbone(
                vision_answer="yes",
                vision_dist=torch.tensor([1.0, 0.0], dtype=torch.float32),
                text_answer="no",
                text_dist=torch.tensor([0.0, 1.0], dtype=torch.float32),
            )
        )

        pred = baseline.predict(_example())

        self.assertEqual(pred.final_answer, "<ABSTAIN>")
        self.assertTrue(pred.abstained)
        self.assertLess(pred.confidence, 0.5)
        self.assertEqual(pred.metadata["vision_raw_output"], "raw::yes")
        self.assertEqual(pred.metadata["text_raw_output"], "raw::no")

    def test_confidence_threshold_abstains_when_inverse_entropy_is_low(self) -> None:
        baseline = ConfidenceThresholdBaseline(
            _ControlledBackbone(mm_answer="yes", mm_dist=torch.tensor([0.5, 0.5], dtype=torch.float32)),
            threshold=0.3,
        )

        pred = baseline.predict(_example())

        self.assertEqual(pred.final_answer, "<ABSTAIN>")
        self.assertTrue(pred.abstained)
        self.assertAlmostEqual(pred.confidence, 0.0, places=6)
        self.assertEqual(pred.raw_text, "raw::yes")

    def test_probe_heuristic_routes_to_lower_entropy_probe(self) -> None:
        baseline = ProbeHeuristicBaseline(
            _ControlledBackbone(
                vision_answer="2",
                vision_dist=torch.tensor([0.95, 0.05], dtype=torch.float32),
                text_answer="3",
                text_dist=torch.tensor([0.55, 0.45], dtype=torch.float32),
            ),
            both_uncertain_threshold=2.0,
        )

        pred = baseline.predict(_example())

        self.assertEqual(pred.final_answer, "2")
        self.assertFalse(pred.abstained)
        self.assertAlmostEqual(pred.confidence, 0.95, places=6)
        self.assertEqual(pred.raw_text, "raw::2")
        self.assertEqual(pred.metadata["vision_raw_output"], "raw::2")
        self.assertEqual(pred.metadata["text_raw_output"], "raw::3")

    def test_probe_heuristic_abstains_when_both_probes_are_uncertain(self) -> None:
        baseline = ProbeHeuristicBaseline(
            _ControlledBackbone(
                vision_answer="2",
                vision_dist=torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32),
                text_answer="3",
                text_dist=torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32),
            ),
            both_uncertain_threshold=1.0,
        )

        pred = baseline.predict(_example())

        self.assertEqual(pred.final_answer, "<ABSTAIN>")
        self.assertTrue(pred.abstained)
        self.assertAlmostEqual(pred.confidence, 0.34, places=6)
        self.assertEqual(pred.raw_text, "raw::2")
        self.assertEqual(pred.metadata["vision_raw_output"], "raw::2")
        self.assertEqual(pred.metadata["text_raw_output"], "raw::3")


if __name__ == "__main__":
    unittest.main()
