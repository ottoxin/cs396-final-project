from __future__ import annotations

import unittest

from carm.data.integrity import IntegrityError, validate_split_integrity
from carm.data.schema import Split
from tests.fixtures import make_examples


class TestIntegrity(unittest.TestCase):
    def test_integrity_manifest(self) -> None:
        examples = make_examples()
        manifest = validate_split_integrity(examples)
        self.assertIn("train", manifest)
        self.assertIn("val", manifest)
        self.assertIn("test", manifest)

    def test_detect_duplicate_id(self) -> None:
        examples = make_examples()
        bad = list(examples)
        bad[1] = bad[1].__class__(**{**bad[1].__dict__, "example_id": bad[0].example_id})
        with self.assertRaises(IntegrityError):
            validate_split_integrity(bad)

    def test_detect_image_leakage(self) -> None:
        examples = make_examples()
        bad = list(examples)
        bad[-1] = bad[-1].__class__(**{**bad[-1].__dict__, "split": Split.TRAIN, "source_image_id": "s2"})
        with self.assertRaises(IntegrityError):
            validate_split_integrity(bad)


if __name__ == "__main__":
    unittest.main()
