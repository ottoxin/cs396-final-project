from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from carm.data.construction import build_conflict_suite
from carm.data.io import load_examples, save_examples
from carm.data.sampling import sample_pilot_by_base
from carm.data.schema import Family
from carm.eval.baselines import BackboneDirectBaseline, TwoPassSelfConsistencyBaseline
from carm.eval.evaluator import evaluate_predictor
from carm.models.backbone import MockFrozenBackbone
from tests.fixtures import make_base_examples


class TestIntegrationPhaseASmoke(unittest.TestCase):
    def test_build_sample_and_baselines(self) -> None:
        base = make_base_examples()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base_jsonl = root / "base.jsonl"
            full_jsonl = root / "full.jsonl"
            pilot_jsonl = root / "pilot.jsonl"
            save_examples(base_jsonl, base)

            loaded_base = load_examples(base_jsonl)
            suite, _ = build_conflict_suite(
                loaded_base,
                seed=7,
                held_out_family=Family.ATTRIBUTE_COLOR,
                held_out_severity=3,
            )
            save_examples(full_jsonl, suite)

            sampled, pilot_manifest = sample_pilot_by_base(suite, base_sample_size=2, seed=7)
            save_examples(pilot_jsonl, sampled)
            self.assertIn("selected_base_count", pilot_manifest)
            self.assertGreater(len(sampled), 0)

            backbone = MockFrozenBackbone()
            eval_out = root / "eval"

            direct_metrics = evaluate_predictor(BackboneDirectBaseline(backbone), sampled, output_dir=eval_out / "direct")
            two_pass_metrics = evaluate_predictor(
                TwoPassSelfConsistencyBaseline(backbone),
                sampled,
                output_dir=eval_out / "two_pass",
            )

            self.assertIn("accuracy", direct_metrics)
            self.assertIn("macro_f1_conflict", two_pass_metrics)

            pred_file = eval_out / "two_pass" / "per_example_predictions.jsonl"
            self.assertTrue(pred_file.exists())
            row = json.loads(pred_file.read_text(encoding="utf-8").splitlines()[0])
            required = {
                "pred_conflict_type",
                "pred_action",
                "r_v",
                "r_t",
                "abstained",
                "final_answer",
                "correct",
            }
            self.assertTrue(required.issubset(set(row.keys())))


if __name__ == "__main__":
    unittest.main()
