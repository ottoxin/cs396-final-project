from __future__ import annotations

import unittest

from scripts.tune_baseline_thresholds import (
    _select_best_candidate,
    _sweep_confidence_threshold,
    _sweep_probe_threshold,
)


class TestTuneBaselineThresholds(unittest.TestCase):
    def test_sweep_confidence_threshold_adds_abstain_all_candidate(self) -> None:
        sweep = _sweep_confidence_threshold(
            [
                {
                    "confidence": 0.2,
                    "would_be_correct": 1.0,
                    "answered_success": 1.0,
                    "abstained_success": 0.0,
                },
                {
                    "confidence": 0.8,
                    "would_be_correct": 0.0,
                    "answered_success": 0.0,
                    "abstained_success": 1.0,
                },
            ]
        )

        self.assertEqual(len(sweep), 3)
        self.assertEqual(sweep[-1]["coverage"], 0.0)
        self.assertEqual(sweep[-1]["accuracy"], 0.0)

    def test_sweep_probe_threshold_adds_abstain_all_candidate(self) -> None:
        sweep = _sweep_probe_threshold(
            [
                {
                    "min_entropy": 0.5,
                    "would_be_correct": 1.0,
                    "answered_success": 1.0,
                    "abstained_success": 0.0,
                },
                {
                    "min_entropy": 1.5,
                    "would_be_correct": 0.0,
                    "answered_success": 0.0,
                    "abstained_success": 1.0,
                },
            ]
        )

        self.assertEqual(len(sweep), 3)
        self.assertEqual(sweep[0]["coverage"], 0.0)
        self.assertEqual(sweep[0]["accuracy"], 0.0)

    def test_select_best_candidate_prefers_lowest_confidence_threshold_on_exact_tie(self) -> None:
        selected = _select_best_candidate(
            [
                {"threshold": 0.6, "task_success": 0.8, "coverage": 0.5, "accuracy": 0.5, "accuracy_on_answered": 1.0},
                {"threshold": 0.4, "task_success": 0.8, "coverage": 0.5, "accuracy": 0.5, "accuracy_on_answered": 1.0},
            ],
            threshold_key="threshold",
            prefer_lower_threshold=True,
        )

        self.assertEqual(selected["threshold"], 0.4)

    def test_select_best_candidate_prefers_highest_probe_threshold_on_exact_tie(self) -> None:
        selected = _select_best_candidate(
            [
                {"threshold": 0.6, "task_success": 0.8, "coverage": 0.5, "accuracy": 0.5, "accuracy_on_answered": 1.0},
                {"threshold": 0.4, "task_success": 0.8, "coverage": 0.5, "accuracy": 0.5, "accuracy_on_answered": 1.0},
            ],
            threshold_key="threshold",
            prefer_lower_threshold=False,
        )

        self.assertEqual(selected["threshold"], 0.6)


if __name__ == "__main__":
    unittest.main()
