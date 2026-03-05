from __future__ import annotations

import unittest

from scripts.run_baselines import _prune_summary


class TestRunBaselines(unittest.TestCase):
    def test_prune_summary_removes_stale_baseline_keys(self) -> None:
        summary = {
            "backbone_direct": {"accuracy": 0.1},
            "two_pass_self_consistency": {"accuracy": 0.2},
            "probe_only_heuristic": {"accuracy": 0.3},
        }
        active = {"backbone_direct", "probe_only_heuristic"}

        pruned, stale = _prune_summary(summary, active)

        self.assertEqual(stale, ["two_pass_self_consistency"])
        self.assertEqual(set(pruned.keys()), active)


if __name__ == "__main__":
    unittest.main()
