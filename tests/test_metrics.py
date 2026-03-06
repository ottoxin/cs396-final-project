from __future__ import annotations

import unittest

from carm.eval.metrics import accuracy_on_answered, risk_coverage_curve_task_success, summarize_metrics, task_success_single


class TestMetrics(unittest.TestCase):
    def test_task_success_single_rules(self) -> None:
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "abstain",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "abstain",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "protocol_category": "C2",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "protocol_category": "C2",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "trust_vision",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "protocol_category": "C2",
                    "pred_action": "require_agreement",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "trust_text",
                    "abstained": True,
                    "correct": True,
                }
            )
        )

    def test_accuracy_on_answered_ignores_abstained_examples(self) -> None:
        records = [
            {"abstained": False, "correct": True},
            {"abstained": False, "correct": False},
            {"abstained": True, "correct": True},
        ]

        self.assertAlmostEqual(accuracy_on_answered(records), 0.5, places=6)

    def test_risk_coverage_task_success_sweeps_confidence_threshold(self) -> None:
        records = [
            {
                "oracle_action": "trust_vision",
                "abstained": False,
                "correct": True,
                "confidence": 0.9,
            },
            {
                "oracle_action": "abstain",
                "abstained": True,
                "correct": False,
                "confidence": 0.2,
            },
        ]

        curve = risk_coverage_curve_task_success(records)

        self.assertGreaterEqual(len(curve), 2)
        self.assertEqual(curve[0]["coverage"], 0.0)
        self.assertAlmostEqual(curve[-1]["coverage"], 0.5, places=6)
        self.assertAlmostEqual(curve[-1]["risk"], 0.0, places=6)

    def test_summarize_metrics_groups_by_category_and_split(self) -> None:
        records = [
            {
                "oracle_action": "require_agreement",
                "abstained": False,
                "correct": True,
                "confidence": 0.9,
                "split": "val",
                "family": "none",
                "protocol_category": "C1",
            },
            {
                "oracle_action": "abstain",
                "abstained": True,
                "correct": False,
                "confidence": 0.8,
                "split": "test_id",
                "family": "none",
                "protocol_category": "C5",
            },
        ]

        metrics = summarize_metrics(records)

        self.assertEqual(metrics["task_success"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["accuracy_on_answered"], 1.0)
        self.assertIn("risk_coverage_task_success", metrics)
        self.assertEqual(metrics["task_success_per_split"]["val"], 1.0)
        self.assertEqual(metrics["task_success_per_split"]["test_id"], 1.0)
        self.assertEqual(metrics["task_success_per_category"]["C1"], 1.0)
        self.assertEqual(metrics["task_success_per_category"]["C5"], 1.0)
        self.assertEqual(metrics["accuracy_per_category"]["C1"], 1.0)
        self.assertEqual(metrics["accuracy_per_category"]["C5"], 0.0)
        self.assertEqual(metrics["example_counts_by_split"]["val"], 1)
        self.assertEqual(metrics["example_counts_by_category"]["C5"], 1)


if __name__ == "__main__":
    unittest.main()
