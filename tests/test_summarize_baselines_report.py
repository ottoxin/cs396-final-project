from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_baselines_report import summarize_baselines_root


class TestSummarizeBaselinesReport(unittest.TestCase):
    def test_summary_writes_c2_diagnostics_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "baselines"
            baseline_dir = root / "probe_heuristic"
            baseline_dir.mkdir(parents=True)
            rows = [
                {
                    "example_id": "ex1",
                    "split": "test_id",
                    "oracle_action": "abstain",
                    "protocol_category": "C2",
                    "final_answer": "<ABSTAIN>",
                    "abstained": True,
                    "confidence": 0.7,
                    "correct": False,
                    "task_success": True,
                    "c2_vision_only_correct": True,
                    "c2_text_only_correct": False,
                    "c2_multimodal_abstained": True,
                }
            ]
            with (baseline_dir / "per_example_predictions.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            outputs = summarize_baselines_root(root, target_coverage=0.8, split_filter="test_id")

            self.assertTrue(outputs["main_md"].exists())
            self.assertTrue(outputs["category_md"].exists())
            self.assertTrue(outputs["c2_md"].exists())
            c2_md = outputs["c2_md"].read_text(encoding="utf-8")
            self.assertIn("vision_only_acc", c2_md)
            self.assertIn("1.0000 (n=1)", c2_md)
            self.assertIn("probe_heuristic", c2_md)


if __name__ == "__main__":
    unittest.main()
