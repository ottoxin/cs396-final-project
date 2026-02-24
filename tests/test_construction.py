from __future__ import annotations

import unittest

from carm.data.construction import build_conflict_suite
from carm.data.schema import Family, Operator, Split
from tests.fixtures import make_base_examples


class TestConstruction(unittest.TestCase):
    def test_required_operators_generated(self) -> None:
        suite, _ = build_conflict_suite(
            make_base_examples(),
            seed=7,
            held_out_family=Family.ATTRIBUTE_COLOR,
            held_out_severity=3,
        )
        ops = {row.operator.value for row in suite}
        self.assertIn(Operator.CLEAN.value, ops)
        self.assertIn(Operator.SWAP_EASY.value, ops)
        self.assertIn(Operator.SWAP_HARD.value, ops)
        self.assertIn(Operator.TEXT_EDIT.value, ops)
        self.assertIn(Operator.VISION_CORRUPT.value, ops)

    def test_ood_family_assignment(self) -> None:
        suite, _ = build_conflict_suite(
            make_base_examples(),
            seed=7,
            held_out_family=Family.ATTRIBUTE_COLOR,
            held_out_severity=3,
        )
        ood_family_rows = [r for r in suite if r.split == Split.TEST_OOD_FAMILY]
        self.assertGreater(len(ood_family_rows), 0)
        self.assertTrue(all(r.family == Family.ATTRIBUTE_COLOR for r in ood_family_rows))


if __name__ == "__main__":
    unittest.main()
