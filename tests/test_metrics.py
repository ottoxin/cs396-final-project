from __future__ import annotations

import unittest

from carm.eval.metrics import risk_coverage_curve_task_success, summarize_metrics, task_success_single


class TestMetrics(unittest.TestCase):
    def test_task_success_single_rules(self) -> None:
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "abstain",
                    "pred_action": "abstain",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "abstain",
                    "pred_action": "abstain",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "pred_action": "require_agreement",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "pred_action": "trust_vision",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "trust_vision",
                    "pred_action": "trust_vision",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "trust_text",
                    "pred_action": "trust_text",
                    "abstained": False,
                    "correct": False,
                }
            )
        )

    def test_risk_coverage_task_success_uses_task_success_outcome(self) -> None:
        records = [
            {
                "oracle_action": "abstain",
                "pred_action": "abstain",
                "abstained": True,
                "correct": False,
                "task_success": True,
                "confidence": 0.9,
            },
            {
                "oracle_action": "trust_vision",
                "pred_action": "trust_vision",
                "abstained": False,
                "correct": True,
                "task_success": False,
                "confidence": 0.8,
            },
        ]
        curve = risk_coverage_curve_task_success(records)
        self.assertEqual(len(curve), 2)
        self.assertEqual(curve[0]["coverage"], 0.5)
        self.assertEqual(curve[0]["risk"], 0.0)

    def test_task_success_per_category(self) -> None:
        records = [
            {
                "oracle_action": "require_agreement",
                "pred_action": "require_agreement",
                "abstained": False,
                "correct": True,
                "confidence": 0.9,
                "split": "val",
                "family": "none",
                "pred_conflict_type": "none",
                "r_v": 0.5,
                "r_t": 0.5,
                "target_r_v": 0.5,
                "target_r_t": 0.5,
                "metadata": {"protocol_category": "C1"},
            },
            {
                "oracle_action": "abstain",
                "pred_action": "abstain",
                "abstained": True,
                "correct": False,
                "confidence": 0.8,
                "split": "val",
                "family": "none",
                "pred_conflict_type": "none",
                "r_v": 0.5,
                "r_t": 0.5,
                "target_r_v": 0.5,
                "target_r_t": 0.5,
                "metadata": {"protocol_category": "C5"},
            },
        ]
        metrics = summarize_metrics(records)
        self.assertEqual(metrics["task_success"], 1.0)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertIn("risk_coverage_task_success", metrics)
        self.assertEqual(metrics["task_success_per_split"]["val"], 1.0)
        self.assertEqual(metrics["task_success_per_category"]["C1"], 1.0)
        self.assertEqual(metrics["task_success_per_category"]["C5"], 1.0)


if __name__ == "__main__":
    unittest.main()
