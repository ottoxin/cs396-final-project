from __future__ import annotations

import unittest

from carm.data.construction import build_conflict_suite
from carm.data.sampling import sample_pilot_by_base
from carm.data.schema import Family
from tests.fixtures import make_base_examples


class TestSampling(unittest.TestCase):
    def test_deterministic_sampling(self) -> None:
        suite, _ = build_conflict_suite(make_base_examples(), seed=7, held_out_family=Family.ATTRIBUTE_COLOR)

        a_rows, a_manifest = sample_pilot_by_base(suite, base_sample_size=2, seed=3)
        b_rows, b_manifest = sample_pilot_by_base(suite, base_sample_size=2, seed=3)

        self.assertEqual([r.example_id for r in a_rows], [r.example_id for r in b_rows])
        self.assertEqual(a_manifest["selected_base_count"], b_manifest["selected_base_count"])

    def test_variant_completeness_by_base(self) -> None:
        suite, _ = build_conflict_suite(make_base_examples(), seed=7, held_out_family=Family.ATTRIBUTE_COLOR)
        sampled, _ = sample_pilot_by_base(suite, base_sample_size=1, seed=7)

        base_ids = {row.base_id for row in sampled}
        self.assertEqual(len(base_ids), 1)

        sampled_base = next(iter(base_ids))
        full_count = sum(1 for row in suite if row.base_id == sampled_base)
        sample_count = sum(1 for row in sampled if row.base_id == sampled_base)
        self.assertEqual(full_count, sample_count)


if __name__ == "__main__":
    unittest.main()
