from __future__ import annotations

import unittest

from carm.data.schema import Action, AnswerType, ConflictExample, CorruptModality, Family, Operator, Split


class TestConflictExampleSchema(unittest.TestCase):
    def test_round_trip_preserves_explicit_c4_targets(self) -> None:
        example = ConflictExample(
            example_id="b1::c4",
            base_id="b1",
            variant_id="c4",
            image_path="image.jpg",
            text_input="caption",
            question="Is there a cat?",
            gold_answer="yes",
            split=Split.VAL,
            family=Family.EXISTENCE,
            operator=Operator.TEXT_EDIT,
            corrupt_modality=CorruptModality.TEXT,
            severity=1,
            answer_type=AnswerType.BOOLEAN,
            oracle_action=Action.ABSTAIN,
            vision_supported_target="yes",
            text_supported_target="no",
            metadata={"protocol_category": "C4"},
        )

        restored = ConflictExample.from_dict(example.to_dict())

        self.assertEqual(restored.oracle_action, Action.ABSTAIN)
        self.assertEqual(restored.vision_supported_target, "yes")
        self.assertEqual(restored.text_supported_target, "no")

    def test_from_dict_normalizes_stale_oracle_action_from_protocol_category(self) -> None:
        restored = ConflictExample.from_dict(
            {
                "example_id": "b1::c4",
                "base_id": "b1",
                "variant_id": "c4",
                "image_path": "image.jpg",
                "text_input": "caption",
                "question": "Is there a cat?",
                "gold_answer": "yes",
                "split": "val",
                "family": "existence",
                "operator": "text_edit",
                "corrupt_modality": "text",
                "severity": 1,
                "answer_type": "boolean",
                "oracle_action": "require_agreement",
                "metadata": {"protocol_category": "C4"},
            }
        )

        self.assertEqual(restored.oracle_action, Action.ABSTAIN)


if __name__ == "__main__":
    unittest.main()
