from __future__ import annotations

import unittest

from carm.eval.metrics import (
    accuracy_on_answered,
    action_accuracy,
    action_macro_f1,
    risk_coverage_curve_task_success,
    summarize_metrics,
    task_success_single,
)


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
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "protocol_category": "C1",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "require_agreement",
                    "protocol_category": "C1",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "trust_vision",
                    "protocol_category": "C2",
                    "vision_info_state": "informative",
                    "text_info_state": "uninformative",
                    "pairwise_relation": "asymmetric",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "trust_vision",
                    "protocol_category": "C2",
                    "vision_info_state": "informative",
                    "text_info_state": "uninformative",
                    "pairwise_relation": "asymmetric",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "trust_text",
                    "protocol_category": "C3",
                    "vision_info_state": "uninformative",
                    "text_info_state": "informative",
                    "pairwise_relation": "asymmetric",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertFalse(
            task_success_single(
                {
                    "oracle_action": "abstain",
                    "protocol_category": "C4",
                    "vision_info_state": "informative",
                    "text_info_state": "informative",
                    "pairwise_relation": "contradictory",
                    "abstained": False,
                    "correct": True,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "abstain",
                    "protocol_category": "C4",
                    "vision_info_state": "informative",
                    "text_info_state": "informative",
                    "pairwise_relation": "contradictory",
                    "abstained": True,
                    "correct": False,
                }
            )
        )
        self.assertTrue(
            task_success_single(
                {
                    "oracle_action": "abstain",
                    "protocol_category": "C5",
                    "abstained": True,
                    "correct": False,
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
                "protocol_category": "C2",
                "vision_info_state": "informative",
                "text_info_state": "uninformative",
                "pairwise_relation": "asymmetric",
                "abstained": False,
                "correct": True,
                "confidence": 0.9,
            },
            {
                "oracle_action": "abstain",
                "protocol_category": "C5",
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

    def test_action_metrics_use_fixed_four_label_space(self) -> None:
        records = [
            {"oracle_action": "trust_vision", "pred_action": "trust_vision"},
            {"oracle_action": "trust_text", "pred_action": "require_agreement"},
            {"oracle_action": "require_agreement", "pred_action": "require_agreement"},
            {"oracle_action": "abstain", "pred_action": "abstain"},
        ]

        self.assertAlmostEqual(action_accuracy(records) or 0.0, 0.75, places=6)
        self.assertAlmostEqual(action_macro_f1(records) or 0.0, 2.0 / 3.0, places=6)

    def test_c4_task_success_is_identical_for_flat_and_action_aware_rows(self) -> None:
        flat = {
            "oracle_action": "abstain",
            "protocol_category": "C4",
            "vision_info_state": "informative",
            "text_info_state": "informative",
            "pairwise_relation": "contradictory",
            "abstained": False,
            "correct": True,
        }
        action_aware = {
            **flat,
            "pred_action": "require_agreement",
        }

        self.assertFalse(task_success_single(flat))
        self.assertFalse(task_success_single(action_aware))

        flat_abstain = dict(flat, abstained=True, correct=False)
        action_abstain = dict(action_aware, abstained=True, correct=False)
        self.assertTrue(task_success_single(flat_abstain))
        self.assertTrue(task_success_single(action_abstain))

    def test_c1_task_success_ignores_pred_action_and_requires_correct_nonabstained_answer(self) -> None:
        row = {
            "oracle_action": "require_agreement",
            "protocol_category": "C1",
            "pred_action": "abstain",
            "abstained": False,
            "correct": True,
        }
        self.assertTrue(task_success_single(row))
        self.assertFalse(task_success_single(dict(row, abstained=True, correct=False)))

    def test_summarize_metrics_groups_by_category_and_split(self) -> None:
        records = [
            {
                "oracle_action": "require_agreement",
                "pred_action": "require_agreement",
                "abstained": False,
                "correct": True,
                "confidence": 0.9,
                "projection_succeeded": True,
                "used_fallback_dist": False,
                "parsed_unknown": False,
                "parsed_in_active_vocab": True,
                "canonicalized_candidate": "yes",
                "out_of_vocab_generation": False,
                "dist_argmax_label": "yes",
                "parsed_argmax_agree": True,
                "final_answer": "yes",
                "split": "val",
                "family": "none",
                "protocol_category": "C1",
            },
            {
                "oracle_action": "abstain",
                "pred_action": "abstain",
                "abstained": True,
                "correct": False,
                "confidence": 0.8,
                "projection_succeeded": False,
                "used_fallback_dist": True,
                "parsed_unknown": True,
                "parsed_in_active_vocab": False,
                "canonicalized_candidate": None,
                "out_of_vocab_generation": True,
                "dist_argmax_label": "unknown",
                "parsed_argmax_agree": True,
                "final_answer": "unknown",
                "split": "test_id",
                "family": "none",
                "protocol_category": "C5",
            },
            {
                "oracle_action": "abstain",
                "pred_action": "abstain",
                "abstained": True,
                "correct": False,
                "confidence": 0.7,
                "projection_succeeded": True,
                "used_fallback_dist": False,
                "parsed_unknown": False,
                "parsed_in_active_vocab": True,
                "canonicalized_candidate": "yes",
                "out_of_vocab_generation": False,
                "dist_argmax_label": "yes",
                "parsed_argmax_agree": True,
                "final_answer": "<ABSTAIN>",
                "split": "val",
                "family": "existence",
                "protocol_category": "C4",
                "vision_info_state": "informative",
                "text_info_state": "informative",
                "pairwise_relation": "contradictory",
                "c2_vision_only_correct": True,
                "c2_text_only_correct": False,
                "c2_multimodal_abstained": True,
            },
        ]

        metrics = summarize_metrics(records)

        self.assertEqual(metrics["task_success"], 1.0)
        self.assertEqual(metrics["action_accuracy"], 1.0)
        self.assertEqual(metrics["action_macro_f1"], 0.5)
        self.assertEqual(metrics["accuracy"], 1.0 / 3.0)
        self.assertEqual(metrics["coverage"], 1.0 / 3.0)
        self.assertEqual(metrics["accuracy_on_answered"], 1.0)
        self.assertIn("risk_coverage_task_success", metrics)
        self.assertEqual(metrics["task_success_per_split"]["val"], 1.0)
        self.assertEqual(metrics["task_success_per_split"]["test_id"], 1.0)
        self.assertEqual(metrics["task_success_per_category"]["C1"], 1.0)
        self.assertEqual(metrics["task_success_per_category"]["C4"], 1.0)
        self.assertEqual(metrics["task_success_per_category"]["C5"], 1.0)
        self.assertEqual(metrics["accuracy_per_category"]["C1"], 1.0)
        self.assertEqual(metrics["accuracy_per_category"]["C5"], 0.0)
        self.assertEqual(metrics["projection_success_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["fallback_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["parsed_unknown_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["out_of_vocab_generation_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["parsed_argmax_agreement_rate"], 1.0)
        self.assertEqual(metrics["final_unknown_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["c2_vision_only_accuracy"], 1.0)
        self.assertEqual(metrics["c2_vision_only_count"], 1)
        self.assertEqual(metrics["c2_text_only_accuracy"], 0.0)
        self.assertEqual(metrics["c2_text_only_count"], 1)
        self.assertEqual(metrics["c2_multimodal_abstention_rate"], 1.0)
        self.assertEqual(metrics["c2_multimodal_abstention_count"], 1)
        self.assertEqual(metrics["projection_success_rate_per_category"]["C1"], 1.0)
        self.assertEqual(metrics["fallback_rate_per_category"]["C5"], 1.0)
        self.assertEqual(metrics["example_counts_by_split"]["val"], 2)
        self.assertEqual(metrics["example_counts_by_category"]["C5"], 1)

    def test_c4_diagnostics_return_none_when_unavailable(self) -> None:
        metrics = summarize_metrics(
            [
                {
                    "oracle_action": "abstain",
                    "abstained": True,
                    "correct": False,
                    "protocol_category": "C4",
                    "vision_info_state": "informative",
                    "text_info_state": "informative",
                    "pairwise_relation": "contradictory",
                    "c2_multimodal_abstained": True,
                }
            ]
        )
        self.assertIsNone(metrics["c2_vision_only_accuracy"])
        self.assertEqual(metrics["c2_vision_only_count"], 0)
        self.assertIsNone(metrics["c2_text_only_accuracy"])
        self.assertEqual(metrics["c2_text_only_count"], 0)
        self.assertEqual(metrics["c2_multimodal_abstention_rate"], 1.0)
        self.assertEqual(metrics["c2_multimodal_abstention_count"], 1)


if __name__ == "__main__":
    unittest.main()
