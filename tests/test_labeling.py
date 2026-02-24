from __future__ import annotations

import unittest

from carm.data.labeling import derive_oracle_action, derive_reliability_target
from carm.data.schema import CorruptModality, EvidenceModality


class TestLabeling(unittest.TestCase):
    def test_oracle_mapping(self) -> None:
        self.assertEqual(derive_oracle_action(CorruptModality.TEXT).value, "trust_vision")
        self.assertEqual(derive_oracle_action(CorruptModality.VISION).value, "trust_text")
        self.assertEqual(derive_oracle_action(CorruptModality.NONE).value, "require_agreement")
        self.assertEqual(derive_oracle_action(CorruptModality.BOTH).value, "abstain")

    def test_reliability_monotonic_with_severity(self) -> None:
        low = derive_reliability_target(EvidenceModality.VISION_REQUIRED, CorruptModality.VISION, severity=1)
        high = derive_reliability_target(EvidenceModality.VISION_REQUIRED, CorruptModality.VISION, severity=3)
        self.assertGreaterEqual(low.r_v, high.r_v)


if __name__ == "__main__":
    unittest.main()
