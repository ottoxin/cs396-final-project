from __future__ import annotations

import unittest

from carm.data.schema import Action
from carm.models.interfaces import ProbeResult
from carm.models.policy import answers_agree, apply_action_and_generate

import torch


class TestPolicy(unittest.TestCase):
    def test_structured_equivalence(self) -> None:
        self.assertTrue(answers_agree("yes", "true"))
        self.assertFalse(answers_agree("left", "right"))

    def test_require_agreement_path(self) -> None:
        v = ProbeResult(answer_dist=torch.tensor([1.0]), answer_text="2", features=torch.zeros(3))
        t = ProbeResult(answer_dist=torch.tensor([1.0]), answer_text="2", features=torch.zeros(3))
        answer, abstained, audit = apply_action_and_generate(Action.REQUIRE_AGREEMENT, v, t)
        self.assertEqual(answer, "2")
        self.assertFalse(abstained)
        self.assertEqual(audit["path"], "require_agreement")


if __name__ == "__main__":
    unittest.main()
