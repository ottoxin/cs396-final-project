from __future__ import annotations

import unittest

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
from carm.experimental.labels import derive_labels


def _make_example(
    *,
    example_id: str,
    image_state: str,
    caption_state: str,
    oracle_action: Action,
    vision_info_state: str | None = None,
    text_info_state: str | None = None,
    pairwise_relation: str | None = None,
    joint_answer: str | None = None,
) -> ConflictExample:
    return ConflictExample(
        example_id=example_id,
        base_id=example_id.split("::")[0],
        variant_id=example_id.split("::")[-1],
        image_path="debug.jpg",
        text_input="A caption.",
        question="Is there a cat?",
        gold_answer="yes",
        split=Split.TRAIN,
        family=Family.EXISTENCE,
        operator=Operator.CLEAN,
        corrupt_modality=CorruptModality.NONE,
        severity=0,
        answer_type=AnswerType.BOOLEAN,
        oracle_action=oracle_action,
        evidence_modality=EvidenceModality.BOTH,
        vision_info_state=vision_info_state,
        text_info_state=text_info_state,
        pairwise_relation=pairwise_relation,
        joint_answer=joint_answer,
        metadata={
            "image_state": image_state,
            "caption_state": caption_state,
            "protocol_category": "TEST",
        },
    )


class TestExperimentalLabels(unittest.TestCase):
    def test_clean_clean_maps_to_successful_require_agreement(self) -> None:
        derived = derive_labels(
            _make_example(
                example_id="c1::clean",
                image_state="clean",
                caption_state="clean",
                oracle_action=Action.REQUIRE_AGREEMENT,
            )
        )

        self.assertEqual(derived.derivation_status, "success")
        self.assertEqual(derived.vision_info_state, "informative")
        self.assertEqual(derived.text_info_state, "informative")
        self.assertEqual(derived.joint_info_state, "vision_informative__text_informative")
        self.assertEqual(derived.pairwise_relation, "consistent")
        self.assertEqual(derived.reliability_proxy_target, "both_high")
        self.assertEqual(derived.action_target, "require_agreement")
        self.assertTrue(derived.action_target_available)

    def test_clean_different_maps_to_contradictory_abstain(self) -> None:
        derived = derive_labels(
            _make_example(
                example_id="c2::different",
                image_state="clean",
                caption_state="different",
                oracle_action=Action.ABSTAIN,
            )
        )

        self.assertEqual(derived.derivation_status, "success")
        self.assertEqual(derived.vision_info_state, "informative")
        self.assertEqual(derived.text_info_state, "informative")
        self.assertEqual(derived.joint_info_state, "vision_informative__text_informative")
        self.assertEqual(derived.pairwise_relation, "contradictory")
        self.assertEqual(derived.reliability_proxy_target, "both_high")
        self.assertEqual(derived.action_target, "abstain")
        self.assertTrue(derived.action_target_available)
        self.assertFalse(derived.metric_semantics_mismatch)

    def test_explicit_structured_c2_fields_make_action_target_available(self) -> None:
        derived = derive_labels(
            _make_example(
                example_id="c2::structured",
                image_state="clean",
                caption_state="different",
                oracle_action=Action.ABSTAIN,
                vision_info_state="informative",
                text_info_state="informative",
                pairwise_relation="contradictory",
                joint_answer="<ABSTAIN>",
            )
        )

        self.assertEqual(derived.derivation_status, "success")
        self.assertEqual(derived.joint_info_state, "vision_informative__text_informative")
        self.assertEqual(derived.pairwise_relation, "contradictory")
        self.assertEqual(derived.reliability_proxy_target, "both_high")
        self.assertEqual(derived.action_target, "abstain")
        self.assertTrue(derived.action_target_available)
        self.assertFalse(derived.metric_semantics_mismatch)


if __name__ == "__main__":
    unittest.main()
