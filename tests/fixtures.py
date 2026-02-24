from __future__ import annotations

from carm.data.labeling import derive_oracle_action
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


def make_base_examples() -> list[ConflictExample]:
    rows = [
        ConflictExample(
            example_id="b1::clean",
            base_id="b1",
            variant_id="clean",
            image_path="images/train/COCO_train2014_000000000001.jpg",
            text_input="A red car is parked by the curb.",
            question="What color is the car?",
            gold_answer="red",
            split=Split.TRAIN,
            family=Family.ATTRIBUTE_COLOR,
            operator=Operator.CLEAN,
            corrupt_modality=CorruptModality.NONE,
            severity=0,
            answer_type=AnswerType.COLOR,
            oracle_action=Action.REQUIRE_AGREEMENT,
            source_image_id="train::1",
            template_id="tmpl_color",
            evidence_modality=EvidenceModality.VISION_REQUIRED,
        ),
        ConflictExample(
            example_id="b2::clean",
            base_id="b2",
            variant_id="clean",
            image_path="images/train/COCO_train2014_000000000002.jpg",
            text_input="Two dogs sit on grass.",
            question="How many dogs are there?",
            gold_answer="2",
            split=Split.TRAIN,
            family=Family.COUNT,
            operator=Operator.CLEAN,
            corrupt_modality=CorruptModality.NONE,
            severity=0,
            answer_type=AnswerType.INTEGER,
            oracle_action=Action.REQUIRE_AGREEMENT,
            source_image_id="train::2",
            template_id="tmpl_count",
            evidence_modality=EvidenceModality.BOTH,
        ),
        ConflictExample(
            example_id="b3::clean",
            base_id="b3",
            variant_id="clean",
            image_path="images/val/COCO_val2014_000000000003.jpg",
            text_input="A bicycle is next to a tree.",
            question="Is there a bicycle in the image?",
            gold_answer="yes",
            split=Split.TRAIN,
            family=Family.EXISTENCE,
            operator=Operator.CLEAN,
            corrupt_modality=CorruptModality.NONE,
            severity=0,
            answer_type=AnswerType.BOOLEAN,
            oracle_action=Action.REQUIRE_AGREEMENT,
            source_image_id="val::3",
            template_id="tmpl_exist",
            evidence_modality=EvidenceModality.EITHER,
        ),
    ]

    for ex in rows:
        ex.oracle_action = derive_oracle_action(ex.corrupt_modality)
    return rows
