from __future__ import annotations

import unittest

from carm.data.hf5way import (
    SplitRatios,
    answer_type_for_family,
    assign_splits_by_base,
    derive_protocol_category,
    normalize_oracle_action,
    schema_fields_for_category,
)


class TestHF5WayHelpers(unittest.TestCase):
    def test_action_alias(self) -> None:
        self.assertEqual(normalize_oracle_action("trust_image"), "trust_vision")
        self.assertEqual(normalize_oracle_action("require_agreement"), "require_agreement")

    def test_protocol_mapping(self) -> None:
        self.assertEqual(
            derive_protocol_category("clean", "clean", "require_agreement"),
            "C1",
        )
        self.assertEqual(
            derive_protocol_category("clean", "different", "require_agreement"),
            "C2",
        )
        self.assertEqual(
            derive_protocol_category("clean", "irrelevant", "trust_image"),
            "C3",
        )
        self.assertEqual(
            derive_protocol_category("irrelevant", "clean", "trust_text"),
            "C4",
        )
        self.assertEqual(
            derive_protocol_category("irrelevant", "irrelevant", "abstain"),
            "C5",
        )

    def test_schema_mapping(self) -> None:
        self.assertEqual(schema_fields_for_category("C1"), ("clean", "none", 0, "require_agreement"))
        self.assertEqual(schema_fields_for_category("C5"), ("both", "both", 1, "abstain"))
        self.assertEqual(answer_type_for_family("existence"), "boolean")
        self.assertEqual(answer_type_for_family("count"), "integer")
        self.assertEqual(answer_type_for_family("attribute_color"), "color")

    def test_split_deterministic(self) -> None:
        rows = []
        for i in range(12):
            base = f"b{i}"
            category = f"C{(i % 5) + 1}"
            family = "existence" if i < 6 else "count"
            for k in range(2):
                rows.append(
                    {
                        "example_id": f"{base}::{k}",
                        "base_id": base,
                        "family": family,
                        "protocol_category": category,
                    }
                )

        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        first = assign_splits_by_base(rows, seed=7, ratios=ratios)
        second = assign_splits_by_base(rows, seed=7, ratios=ratios)
        self.assertEqual(first, second)
        self.assertTrue({"train", "val", "test_id"}.issuperset(set(first.values())))
        self.assertGreater(sum(1 for s in first.values() if s == "test_id"), 0)


if __name__ == "__main__":
    unittest.main()
