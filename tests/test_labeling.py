from __future__ import annotations

import unittest

from carm.data.labeling import derive_oracle_action, derive_reliability_target
from carm.data.schema import CorruptedModality, EvidenceModality


class TestLabeling(unittest.TestCase):
    def test_oracle_mapping(self) -> None:
        action = derive_oracle_action(EvidenceModality.VISION_REQUIRED, CorruptedModality.TEXT)
        self.assertEqual(action.value, "trust_vision")

        action2 = derive_oracle_action(EvidenceModality.BOTH, CorruptedModality.VISION)
        self.assertEqual(action2.value, "abstain")

    def test_reliability_monotonic_with_severity(self) -> None:
        low = derive_reliability_target(EvidenceModality.VISION_REQUIRED, CorruptedModality.VISION, severity=1)
        high = derive_reliability_target(EvidenceModality.VISION_REQUIRED, CorruptedModality.VISION, severity=3)
        self.assertGreaterEqual(low.r_v, high.r_v)


if __name__ == "__main__":
    unittest.main()
