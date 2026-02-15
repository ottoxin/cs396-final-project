from __future__ import annotations

import unittest

import torch

from carm.data.schema import CorruptedModality
from carm.train.losses import counterfactual_hinge


class TestLosses(unittest.TestCase):
    def test_counterfactual_hinge_direction(self) -> None:
        clean = torch.tensor([0.9, 0.8])
        corrupted = torch.tensor([0.2, 0.8])
        loss_ok = counterfactual_hinge(clean, corrupted, CorruptedModality.VISION, margin=0.2)
        self.assertAlmostEqual(float(loss_ok.item()), 0.0, places=6)

        corrupted_bad = torch.tensor([0.85, 0.8])
        loss_bad = counterfactual_hinge(clean, corrupted_bad, CorruptedModality.VISION, margin=0.2)
        self.assertGreater(float(loss_bad.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
