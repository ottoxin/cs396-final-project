from __future__ import annotations

import unittest

from carm.data.schema import Action, Family
from carm.models.interfaces import ProbeResult
from carm.models.policy import answers_agree, apply_action_and_generate

import torch


class TestPolicy(unittest.TestCase):
    def test_structured_equivalence(self) -> None:
        self.assertTrue(answers_agree("yes", "yes"))
        self.assertTrue(answers_agree("yes", "true"))
        self.assertFalse(answers_agree("left", "right"))
        self.assertTrue(answers_agree("green", "green", family=Family.ATTRIBUTE_COLOR))
        self.assertTrue(answers_agree("grey", "gray", family=Family.ATTRIBUTE_COLOR))
        self.assertTrue(answers_agree("2", "two", family=Family.COUNT))
        self.assertTrue(answers_agree("02", "2", family=Family.COUNT))
        self.assertFalse(answers_agree("2", "3", family=Family.COUNT))
        self.assertTrue(answers_agree("custom free form", "custom free form", family=Family.NONE))

    def test_require_agreement_path(self) -> None:
        v = ProbeResult(answer_dist=torch.tensor([1.0]), answer_text="2", features=torch.zeros(3))
        t = ProbeResult(answer_dist=torch.tensor([1.0]), answer_text="2", features=torch.zeros(3))
        answer, abstained, audit = apply_action_and_generate(
            Action.REQUIRE_AGREEMENT,
            v,
            t,
            family=Family.COUNT,
        )
        self.assertEqual(answer, "2")
        self.assertFalse(abstained)
        self.assertEqual(audit["path"], "require_agreement")

    def test_require_agreement_path_returns_canonical_color_alias(self) -> None:
        v = ProbeResult(answer_dist=torch.tensor([1.0]), answer_text="grey", features=torch.zeros(3))
        t = ProbeResult(answer_dist=torch.tensor([1.0]), answer_text="gray", features=torch.zeros(3))
        answer, abstained, audit = apply_action_and_generate(
            Action.REQUIRE_AGREEMENT,
            v,
            t,
            family=Family.ATTRIBUTE_COLOR,
        )
        self.assertEqual(answer, "gray")
        self.assertFalse(abstained)
        self.assertEqual(audit["path"], "require_agreement")


if __name__ == "__main__":
    unittest.main()
