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
from carm.train.losses import ACTION_TO_IDX, build_targets, counterfactual_hinge


class TestLosses(unittest.TestCase):
    def test_counterfactual_hinge_direction(self) -> None:
        clean = torch.tensor([0.9, 0.8])
        corrupted = torch.tensor([0.2, 0.8])
        loss_ok = counterfactual_hinge(clean, corrupted, CorruptModality.VISION, margin=0.2)
        self.assertAlmostEqual(float(loss_ok.item()), 0.0, places=6)

        corrupted_bad = torch.tensor([0.85, 0.8])
        loss_bad = counterfactual_hinge(clean, corrupted_bad, CorruptModality.VISION, margin=0.2)
        self.assertGreater(float(loss_bad.item()), 0.0)

    def test_build_targets_uses_updated_c4_abstain_label_and_fixed_order(self) -> None:
        example = ConflictExample(
            example_id="c4",
            base_id="c4",
            variant_id="clean",
            image_path="img.jpg",
            text_input="caption",
            question="question",
            gold_answer="yes",
            split=Split.VAL,
            family=Family.EXISTENCE,
            operator=Operator.TEXT_EDIT,
            corrupt_modality=CorruptModality.TEXT,
            severity=1,
            answer_type=AnswerType.BOOLEAN,
            oracle_action=Action.ABSTAIN,
            evidence_modality=EvidenceModality.BOTH,
            metadata={"protocol_category": "C4", "c2_text_supported_answer": "no"},
        )

        targets = build_targets(example, device=torch.device("cpu"))
        self.assertEqual(targets.action_idx, ACTION_TO_IDX[Action.ABSTAIN])
        self.assertEqual(ACTION_TO_IDX[Action.REQUIRE_AGREEMENT], 2)
        self.assertEqual(ACTION_TO_IDX[Action.ABSTAIN], 3)


if __name__ == "__main__":
    unittest.main()
