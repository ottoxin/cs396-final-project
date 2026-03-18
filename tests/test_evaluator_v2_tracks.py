from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from carm.data.schema import ConflictExample
from carm.eval.evaluator import evaluate_predictor
from carm.eval.types import AnswerOutput, PolicyOutput, PredictionOutput
from carm.models.interfaces import BackboneResult, ProbeResult
from tests.fixtures import make_base_examples


class _AnswerOnlyPredictor:
    name = "answer_only"

    def predict_answer(self, ex: ConflictExample) -> AnswerOutput:
        return AnswerOutput(raw_text=ex.gold_answer, answer_confidence=0.9, confidence_source="model")

    def predict_policy(self, ex: ConflictExample):
        return None


class _AnswerOnlyPredictorAlt(_AnswerOnlyPredictor):
    name = "answer_only_alt"


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


class _DiagnosticBackbone:
    name = "diagnostic_backbone"

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        return BackboneResult(
            hidden_states=torch.zeros(1, 1),
            answer_dist=torch.tensor([0.9, 0.1], dtype=torch.float32),
            answer_text="yes",
            raw_text="raw::yes",
        )

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        return ProbeResult(
            answer_dist=torch.tensor([0.9, 0.1], dtype=torch.float32),
            answer_text="yes",
            features=torch.zeros(3),
            raw_text="raw::yes",
        )

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        return ProbeResult(
            answer_dist=torch.tensor([0.1, 0.9], dtype=torch.float32),
            answer_text="no",
            features=torch.zeros(3),
            raw_text="raw::no",
        )


class _BackboneBackedPredictor:
    name = "backbone_backed"

    def __init__(self) -> None:
        self.backbone = _DiagnosticBackbone()

    def predict(self, ex: ConflictExample) -> PredictionOutput:
        return PredictionOutput(
            final_answer="<ABSTAIN>",
            abstained=True,
            confidence=0.9,
            raw_text="raw::abstain",
        )


class TestEvaluatorCompatibility(unittest.TestCase):
    @staticmethod
    def _fingerprint_kwargs(**overrides):
        base = {
            "resolved_config_hash": "cfg-hash",
            "selected_split": "val",
            "dataset_manifest_hash": "manifest-hash",
            "git_commit": "git-hash",
        }
        base.update(overrides)
        return base

    def test_answer_only_predictor_is_flattened(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "answer"
            metrics = evaluate_predictor(
                _AnswerOnlyPredictor(),
                examples,
                out,
                track="answer",
                **self._fingerprint_kwargs(),
            )
            self.assertIn("accuracy", metrics)
            self.assertIn("coverage", metrics)
            self.assertTrue((out / "per_example_predictions.jsonl").exists())
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "run_metadata.json").exists())
            row = json.loads((out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["raw_output"], examples[0].gold_answer)
            self.assertIn("projection_succeeded", row)
            self.assertIn("parsed_argmax_agree", row)
            self.assertIn("c2_vision_only_correct", row)
            self.assertIn("c2_text_only_correct", row)
            self.assertIn("c2_multimodal_abstained", row)
            self.assertIsNone(row["c2_vision_only_correct"])
            self.assertIsNone(row["c2_text_only_correct"])
            self.assertIsNone(row["c2_multimodal_abstained"])

    def test_policy_style_predictor_is_flattened(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "policy"
            metrics = evaluate_predictor(
                _PolicyPredictor(),
                examples,
                out,
                track="policy",
                **self._fingerprint_kwargs(),
            )
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
            self.assertIsNone(row["c2_vision_only_correct"])
            self.assertIsNone(row["c2_text_only_correct"])
            self.assertIsNone(row["c2_multimodal_abstained"])

    def test_backbone_backed_predictor_populates_contradiction_diagnostics(self) -> None:
        example = make_base_examples()[2]
        example.metadata = {
            "protocol_category": "C4",
            "c2_text_supported_answer": "yes",
        }
        example.vision_supported_target = "yes"
        example.text_supported_target = "no"
        example.vision_info_state = "informative"
        example.text_info_state = "informative"
        example.pairwise_relation = "contradictory"
        example.joint_answer = "<ABSTAIN>"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "c4"
            metrics = evaluate_predictor(
                _BackboneBackedPredictor(),
                [example],
                out,
                track="all",
                **self._fingerprint_kwargs(),
            )
            self.assertEqual(metrics["c2_vision_only_accuracy"], 1.0)
            self.assertEqual(metrics["c2_text_only_accuracy"], 1.0)
            self.assertEqual(metrics["c2_multimodal_abstention_rate"], 1.0)
            row = json.loads((out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["c2_vision_only_correct"])
            self.assertTrue(row["c2_text_only_correct"])
            self.assertTrue(row["c2_multimodal_abstained"])
            self.assertEqual(row["vision_supported_target"], "yes")
            self.assertEqual(row["text_supported_target"], "no")

    def test_backbone_backed_predictor_uses_legacy_metadata_fallback_for_contradiction_text_target(self) -> None:
        example = make_base_examples()[2]
        example.metadata = {
            "protocol_category": "C4",
            "c2_text_supported_answer": "no",
        }
        example.vision_info_state = "informative"
        example.text_info_state = "informative"
        example.pairwise_relation = "contradictory"
        example.joint_answer = "<ABSTAIN>"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "c4_legacy"
            metrics = evaluate_predictor(
                _BackboneBackedPredictor(),
                [example],
                out,
                track="all",
                **self._fingerprint_kwargs(),
            )
            self.assertEqual(metrics["c2_text_only_accuracy"], 1.0)
            row = json.loads((out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["c2_text_only_correct"])
            self.assertEqual(row["text_supported_target"], "no")

    def test_backbone_backed_predictor_keeps_partial_contradiction_diagnostics_when_text_target_missing(self) -> None:
        example = make_base_examples()[2]
        example.metadata = {"protocol_category": "C4"}
        example.vision_supported_target = "yes"
        example.text_supported_target = None
        example.vision_info_state = "informative"
        example.text_info_state = "informative"
        example.pairwise_relation = "contradictory"
        example.joint_answer = "<ABSTAIN>"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "c4_partial"
            metrics = evaluate_predictor(
                _BackboneBackedPredictor(),
                [example],
                out,
                track="all",
                **self._fingerprint_kwargs(),
            )
            self.assertEqual(metrics["c2_vision_only_accuracy"], 1.0)
            self.assertEqual(metrics["c2_vision_only_count"], 1)
            self.assertIsNone(metrics["c2_text_only_accuracy"])
            self.assertEqual(metrics["c2_text_only_count"], 0)
            self.assertEqual(metrics["c2_multimodal_abstention_rate"], 1.0)
            self.assertEqual(metrics["c2_multimodal_abstention_count"], 1)
            row = json.loads((out / "per_example_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["c2_vision_only_correct"])
            self.assertIsNone(row["c2_text_only_correct"])
            self.assertTrue(row["c2_multimodal_abstained"])

    def test_resume_requires_matching_run_fingerprint(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "resume"
            base_kwargs = self._fingerprint_kwargs()
            evaluate_predictor(_AnswerOnlyPredictor(), examples, out, track="answer", **base_kwargs)
            metrics = evaluate_predictor(
                _AnswerOnlyPredictor(),
                examples,
                out,
                track="answer",
                resume=True,
                **base_kwargs,
            )
            self.assertIn("accuracy", metrics)

    def test_resume_rejects_mismatched_fingerprint_fields(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "resume_mismatch"
            base_kwargs = self._fingerprint_kwargs()
            evaluate_predictor(_AnswerOnlyPredictor(), examples, out, track="answer", **base_kwargs)

            mismatch_cases = {
                "resolved_config_hash": (base_kwargs | {"resolved_config_hash": "cfg-other"}, _AnswerOnlyPredictor()),
                "selected_split": (base_kwargs | {"selected_split": "test_id"}, _AnswerOnlyPredictor()),
                "dataset_manifest_hash": (base_kwargs | {"dataset_manifest_hash": "manifest-other"}, _AnswerOnlyPredictor()),
                "git_commit": (base_kwargs | {"git_commit": "git-other"}, _AnswerOnlyPredictor()),
                "predictor_name": (base_kwargs, _AnswerOnlyPredictorAlt()),
            }

            for label, (kwargs, predictor) in mismatch_cases.items():
                with self.subTest(field=label):
                    with self.assertRaises(RuntimeError):
                        evaluate_predictor(
                            predictor,
                            examples,
                            out,
                            track="answer",
                            resume=True,
                            **kwargs,
                        )

    def test_resume_requires_fingerprint_inputs_or_explicit_unsafe_override(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "resume_missing"
            evaluate_predictor(_AnswerOnlyPredictor(), examples, out, track="answer")

            with self.assertRaises(RuntimeError):
                evaluate_predictor(_AnswerOnlyPredictor(), examples, out, track="answer", resume=True)

            metrics = evaluate_predictor(
                _AnswerOnlyPredictor(),
                examples,
                out,
                track="answer",
                resume=True,
                unsafe_resume_override=True,
            )
            self.assertIn("accuracy", metrics)

    def test_unsafe_resume_override_allows_fingerprint_mismatch(self) -> None:
        examples = make_base_examples()[:1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "resume_unsafe"
            evaluate_predictor(_AnswerOnlyPredictor(), examples, out, track="answer", **self._fingerprint_kwargs())

            evaluate_predictor(
                _AnswerOnlyPredictor(),
                examples,
                out,
                track="answer",
                resume=True,
                unsafe_resume_override=True,
                **self._fingerprint_kwargs(git_commit="git-other"),
            )

            metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["unsafe_resume_override"])
            self.assertIn("fingerprint_mismatch", metadata["override_reason"])


if __name__ == "__main__":
    unittest.main()
