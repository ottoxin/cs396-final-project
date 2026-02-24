from __future__ import annotations

import unittest
from dataclasses import replace

from carm.data.construction import build_conflict_suite
from carm.data.integrity import IntegrityError, validate_split_integrity
from carm.data.schema import Family, Split
from tests.fixtures import make_base_examples


class TestIntegrity(unittest.TestCase):
    def test_integrity_manifest(self) -> None:
        suite, manifest = build_conflict_suite(
            make_base_examples(),
            seed=7,
            held_out_family=Family.ATTRIBUTE_COLOR,
            held_out_severity=3,
        )
        self.assertGreater(len(suite), 0)
        self.assertIn("hashes", manifest)
        self.assertIn("counts", manifest)

    def test_detect_duplicate_id(self) -> None:
        suite, _ = build_conflict_suite(make_base_examples(), seed=7)
        bad = list(suite)
        bad[1] = replace(bad[1], example_id=bad[0].example_id)
        with self.assertRaises(IntegrityError):
            validate_split_integrity(bad, heldout_family=Family.ATTRIBUTE_COLOR, heldout_severity=3)

    def test_detect_image_leakage(self) -> None:
        suite, _ = build_conflict_suite(make_base_examples(), seed=7)
        left = next(ex for ex in suite if ex.split == Split.TRAIN)
        right_idx = next(
            i
            for i, ex in enumerate(suite)
            if ex.split in {Split.VAL, Split.TEST_ID} and ex.split != Split.TRAIN
        )
        bad = list(suite)
        bad[right_idx] = replace(bad[right_idx], source_image_id=left.source_image_id)
        with self.assertRaises(IntegrityError):
            validate_split_integrity(bad, heldout_family=Family.ATTRIBUTE_COLOR, heldout_severity=3)

    def test_template_leakage_check_is_optional(self) -> None:
        suite, _ = build_conflict_suite(make_base_examples(), seed=7)
        left = next(ex for ex in suite if ex.split == Split.TRAIN)
        right_idx = next(i for i, ex in enumerate(suite) if ex.split == Split.TEST_ID)
        bad = list(suite)
        bad[right_idx] = replace(bad[right_idx], template_id=left.template_id)

        # Default behavior keeps template disjointness optional.
        validate_split_integrity(
            bad,
            heldout_family=Family.ATTRIBUTE_COLOR,
            heldout_severity=3,
            enforce_template_disjointness=False,
        )

        with self.assertRaises(IntegrityError):
            validate_split_integrity(
                bad,
                heldout_family=Family.ATTRIBUTE_COLOR,
                heldout_severity=3,
                enforce_template_disjointness=True,
            )


if __name__ == "__main__":
    unittest.main()
