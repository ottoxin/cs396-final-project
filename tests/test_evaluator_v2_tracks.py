from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from carm.data.schema import ConflictExample
from carm.eval.evaluator import evaluate_predictor
from carm.eval.types import AnswerOutput, PolicyOutput
from tests.fixtures import make_base_examples


class _AnswerOnlyPredictor:
    name = "answer_only"

    def predict_answer(self, ex: ConflictExample) -> AnswerOutput:
        return AnswerOutput(raw_text=ex.gold_answer, answer_confidence=0.9, confidence_source="model")

    def predict_policy(self, ex: ConflictExample):
        return None


class _PolicyPredictor:
    name = "policy_predictor"

    def predict_answer(self, ex: ConflictExample) -> AnswerOutput:
        return AnswerOutput(raw_text=ex.gold_answer, answer_confidence=0.8, confidence_source="model")

    def predict_policy(self, ex: ConflictExample) -> PolicyOutput | None:
        return PolicyOutput(
            pred_conflict_type=ex.family.value,
            pred_action=ex.oracle_action.value,
            abstained=False,
            r_v=0.8,
            r_t=0.6,
            policy_confidence=0.8,
            confidence_source="model",
            audit={"path": "unit"},
        )


class TestEvaluatorCompatibility(unittest.TestCase):
    def test_answer_only_predictor_is_flattened(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "answer"
            metrics = evaluate_predictor(_AnswerOnlyPredictor(), examples, out, track="answer")
            self.assertIn("accuracy", metrics)
            self.assertIn("coverage", metrics)
            self.assertTrue((out / "per_example_predictions.jsonl").exists())
            self.assertTrue((out / "metrics.json").exists())
            row = json.loads((out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["raw_output"], examples[0].gold_answer)
            self.assertIn("projection_succeeded", row)
            self.assertIn("parsed_argmax_agree", row)

    def test_policy_style_predictor_is_flattened(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "policy"
            metrics = evaluate_predictor(_PolicyPredictor(), examples, out, track="policy")
            self.assertIn("task_success", metrics)
            self.assertIn("accuracy_on_answered", metrics)
            self.assertTrue((out / "per_example_predictions.jsonl").exists())
            self.assertTrue((out / "metrics.json").exists())
            row = json.loads((out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["pred_action"], examples[0].oracle_action.value)
            self.assertEqual(row["pred_conflict_type"], examples[0].family.value)
            self.assertEqual(row["r_v"], 0.8)
            self.assertEqual(row["r_t"], 0.6)
            self.assertEqual(row["audit"]["path"], "unit")
            self.assertEqual(row["raw_output"], examples[0].gold_answer)
            self.assertIn("projection_succeeded", row)
            self.assertIn("parsed_argmax_agree", row)


if __name__ == "__main__":
    unittest.main()
