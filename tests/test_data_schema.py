from __future__ import annotations

import unittest

from carm.data.schema import ConflictExample
from tests.fixtures import make_base_examples


class TestDataSchema(unittest.TestCase):
    def test_roundtrip_new_fields(self) -> None:
        ex = make_base_examples()[0]
        row = ex.to_dict()
        parsed = ConflictExample.from_dict(row)

        self.assertEqual(parsed.base_id, ex.base_id)
        self.assertEqual(parsed.variant_id, ex.variant_id)
        self.assertEqual(parsed.family.value, ex.family.value)
        self.assertEqual(parsed.operator.value, ex.operator.value)
        self.assertEqual(parsed.corrupt_modality.value, ex.corrupt_modality.value)
        self.assertEqual(parsed.answer_type.value, ex.answer_type.value)
        self.assertEqual(parsed.split.value, ex.split.value)

    def test_backward_alias_parse(self) -> None:
        row = {
            "example_id": "legacy::row",
            "image_path": "images/s1.jpg",
            "text_input": "A red car.",
            "question": "What color is the car?",
            "gold_answer": "red",
            "split": "test",
            "conflict_type": "attribute",
            "corrupted_modality": "text",
            "corruption_family": "text_edit_attribute",
            "severity": 1,
            "oracle_action": "trust_vision",
        }
        parsed = ConflictExample.from_dict(row)
        self.assertEqual(parsed.family.value, "attribute_color")
        self.assertEqual(parsed.operator.value, "text_edit")
        self.assertEqual(parsed.corrupt_modality.value, "text")
        self.assertEqual(parsed.split.value, "test_id")


if __name__ == "__main__":
    unittest.main()
