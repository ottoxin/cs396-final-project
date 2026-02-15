from __future__ import annotations

import unittest

from carm.data.schema import ConflictExample
from tests.fixtures import make_examples


class TestDataSchema(unittest.TestCase):
    def test_roundtrip(self) -> None:
        ex = make_examples()[0]
        row = ex.to_dict()
        parsed = ConflictExample.from_dict(row)
        self.assertEqual(parsed.example_id, ex.example_id)
        self.assertEqual(parsed.oracle_action.value, ex.oracle_action.value)
        self.assertEqual(parsed.split.value, ex.split.value)


if __name__ == "__main__":
    unittest.main()
