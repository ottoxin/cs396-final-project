from __future__ import annotations

import unittest

from carm.data.schema import AnswerType
from carm.eval.canonicalization import CanonicalizationConfig, canonicalize_answer


class TestCanonicalization(unittest.TestCase):
    def test_boolean_synonym_mapping(self) -> None:
        out = canonicalize_answer("true", AnswerType.BOOLEAN)
        self.assertEqual(out.canonical_label, "yes")
        self.assertEqual(out.canonical_status, "mapped")

    def test_count_parsing_and_bounds(self) -> None:
        cfg = CanonicalizationConfig(count_min=0, count_max=20)
        ok = canonicalize_answer("There are three dogs", AnswerType.INTEGER, cfg=cfg)
        bad = canonicalize_answer("31", AnswerType.INTEGER, cfg=cfg)
        self.assertEqual(ok.canonical_label, "3")
        self.assertEqual(ok.canonical_status, "mapped")
        self.assertIsNone(bad.canonical_label)
        self.assertEqual(bad.canonical_status, "unmapped")

    def test_color_synonym_mapping(self) -> None:
        cfg = CanonicalizationConfig(color_synonyms={"grey": "gray"})
        out = canonicalize_answer("grey", AnswerType.COLOR, cfg=cfg)
        self.assertEqual(out.canonical_label, "gray")
        self.assertEqual(out.canonical_status, "mapped")

    def test_unmapped_and_invalid_cases(self) -> None:
        unmapped = canonicalize_answer("striped", AnswerType.COLOR)
        invalid = canonicalize_answer("<ABSTAIN>", AnswerType.COLOR)
        self.assertEqual(unmapped.canonical_status, "unmapped")
        self.assertEqual(invalid.canonical_status, "invalid")


if __name__ == "__main__":
    unittest.main()
