from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from carm.data.answer_vocab import (
    build_family_vocabs,
    canonicalize_candidate_answer,
    load_family_vocabs,
    normalize_gold_answer,
    parse_generated_answer,
    save_family_vocabs,
)
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


def _example(*, example_id: str, split: Split, family: Family, gold_answer: str) -> ConflictExample:
    return ConflictExample(
        example_id=example_id,
        base_id=example_id,
        variant_id="clean",
        image_path="img.jpg",
        text_input="caption",
        question="What color is the bus?" if family == Family.ATTRIBUTE_COLOR else "How many dogs are there?",
        gold_answer=gold_answer,
        split=split,
        family=family,
        operator=Operator.CLEAN,
        corrupt_modality=CorruptModality.NONE,
        severity=0,
        answer_type=AnswerType.COLOR if family == Family.ATTRIBUTE_COLOR else AnswerType.INTEGER,
        oracle_action=Action.TRUST_VISION,
        evidence_modality=EvidenceModality.EITHER,
        metadata={"protocol_category": "C3"},
    )


class TestAnswerVocab(unittest.TestCase):
    def test_canonicalize_candidate_answer_handles_yes_no_and_count(self) -> None:
        self.assertEqual(canonicalize_candidate_answer("true", Family.EXISTENCE), "yes")
        self.assertEqual(canonicalize_candidate_answer("02", Family.COUNT), "2")
        self.assertEqual(canonicalize_candidate_answer("violet", Family.ATTRIBUTE_COLOR), "purple")
        self.assertIsNone(canonicalize_candidate_answer("maybe", Family.EXISTENCE))

    def test_normalize_gold_answer_is_color_permissive(self) -> None:
        self.assertEqual(normalize_gold_answer("Grey", Family.ATTRIBUTE_COLOR), "gray")
        self.assertEqual(normalize_gold_answer("beige", Family.ATTRIBUTE_COLOR), "beige")

    def test_parse_generated_answer_extracts_sentence_form_candidates(self) -> None:
        existence = parse_generated_answer("Yes, there is a bicycle.", Family.EXISTENCE)
        self.assertEqual(existence.candidate_text, "yes")
        self.assertEqual(existence.canonicalized_candidate, "yes")

        count = parse_generated_answer("There are two dogs.", Family.COUNT)
        self.assertEqual(count.candidate_text, "two")
        self.assertEqual(count.canonicalized_candidate, "2")

        color = parse_generated_answer(
            "The wall looks green to me.",
            Family.ATTRIBUTE_COLOR,
            recognized_color_labels={"green", "blue"},
        )
        self.assertEqual(color.candidate_text, "green")
        self.assertEqual(color.canonicalized_candidate, "green")

        oov_color = parse_generated_answer(
            "The paint looks beige.",
            Family.ATTRIBUTE_COLOR,
            recognized_color_labels={"green", "blue"},
        )
        self.assertEqual(oov_color.candidate_text, None)
        self.assertIsNone(oov_color.canonicalized_candidate)

    def test_build_save_and_load_family_vocabs(self) -> None:
        examples = [
            _example(example_id="c1", split=Split.TRAIN, family=Family.COUNT, gold_answer="two"),
            _example(example_id="c2", split=Split.TRAIN, family=Family.COUNT, gold_answer="01"),
            _example(example_id="k1", split=Split.TRAIN, family=Family.ATTRIBUTE_COLOR, gold_answer="Grey"),
            _example(example_id="k2", split=Split.TRAIN, family=Family.ATTRIBUTE_COLOR, gold_answer="beige"),
            _example(example_id="heldout", split=Split.VAL, family=Family.ATTRIBUTE_COLOR, gold_answer="red"),
        ]

        vocabs = build_family_vocabs(examples)

        self.assertEqual(vocabs[Family.EXISTENCE], ("yes", "no", "unknown"))
        self.assertEqual(vocabs[Family.COUNT], ("1", "2", "unknown"))
        self.assertEqual(vocabs[Family.ATTRIBUTE_COLOR], ("beige", "gray", "unknown"))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "family_vocab.json"
            save_family_vocabs(vocabs, path)
            loaded = load_family_vocabs(path)

        self.assertEqual(loaded, vocabs)


if __name__ == "__main__":
    unittest.main()
