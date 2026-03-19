from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from carm.data.schema import Action, ConflictExample
from carm.eval.baselines import DIAGNOSTIC_BOOL_KEYS, DIAGNOSTIC_VALUE_KEYS
from carm.eval.canonicalization import CanonicalizationConfig, canonicalize_answer
from carm.eval.metrics import task_success_from_components
from carm.eval.types import PredictionOutput
from carm.experimental.labels import DerivedLabels, joint_info_state_label
from carm.experimental.model import (
    CascadeCARMHeads,
    DistributionCARMHeads,
    ExperimentalCARMHeads,
    decode_action,
    decode_info_state,
    decode_pairwise_relation,
)
from carm.models.features import entropy, extract_cross_modal_features, extract_cross_modal_features_augmented, top_margin
from carm.models.interfaces import BackboneAdapter
from carm.models.policy import apply_action_and_generate


@dataclass
class PredictorDiagnostics:
    prediction: PredictionOutput
    multimodal_answer: str
    multimodal_confidence: float
    multimodal_entropy: float
    multimodal_margin: float
    vision_answer: str
    vision_confidence: float
    vision_entropy: float
    vision_margin: float
    text_answer: str
    text_confidence: float
    text_entropy: float
    text_margin: float
    multimodal_raw_output: str | None
    vision_raw_output: str | None
    text_raw_output: str | None


def _as_prediction_output(prediction: Any) -> PredictionOutput:
    if isinstance(prediction, PredictionOutput):
        return prediction
    return PredictionOutput(
        final_answer=str(getattr(prediction, "final_answer", "")),
        abstained=bool(getattr(prediction, "abstained", False)),
        confidence=float(getattr(prediction, "confidence", 0.0) or 0.0),
        metadata=dict(getattr(prediction, "metadata", {}) or {}),
        raw_text=getattr(prediction, "raw_text", None),
    )


def _result_metadata(result: Any) -> dict[str, Any]:
    return dict(getattr(result, "metadata", None) or {})


def _prediction_confidence(dist: torch.Tensor) -> float:
    return float(torch.max(dist).item()) if dist.numel() else 0.0


def _dist_entropy(dist: torch.Tensor) -> float:
    return float(entropy(dist).item()) if dist.numel() else 0.0


def _dist_margin(dist: torch.Tensor) -> float:
    return float(top_margin(dist).item()) if dist.numel() else 0.0


def _answers_match(pred_text: str, gold_text: str, example: ConflictExample, cfg: CanonicalizationConfig) -> bool:
    pred = canonicalize_answer(pred_text, example.answer_type, cfg=cfg)
    gold = canonicalize_answer(gold_text, example.answer_type, cfg=cfg)
    if pred.canonical_label is not None and gold.canonical_label is not None:
        return pred.canonical_label == gold.canonical_label
    return bool(pred.normalized_text) and pred.normalized_text == gold.normalized_text


def _effective_protocol_category(example: ConflictExample) -> str:
    if example.pairwise_relation == "consistent":
        return "C1"
    if example.pairwise_relation == "contradictory":
        return "C4"
    if example.pairwise_relation == "both_weak":
        return "C5"
    if example.pairwise_relation == "asymmetric":
        if example.vision_info_state == "informative" and example.text_info_state == "uninformative":
            return "C2"
        if example.vision_info_state == "uninformative" and example.text_info_state == "informative":
            return "C3"
    if isinstance(example.metadata, dict):
        raw = str(example.metadata.get("protocol_category", "")).strip().upper()
        legacy_to_hf = {
            "C1": "C1",
            "C2": "C4",
            "C3": "C2",
            "C4": "C3",
            "C5": "C5",
        }
        return legacy_to_hf.get(raw, raw)
    return ""


def _revised_task_success(action_target: str | None, *, abstained: bool, correct: bool) -> bool | None:
    if action_target is None:
        return None
    if action_target == Action.ABSTAIN.value:
        return bool(abstained)
    return (not bool(abstained)) and bool(correct)


def _confidence_threshold_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    comparable = [row for row in rows if row.get("task_success_revised") is not None]
    if not comparable:
        return []

    thresholds = sorted({float(row.get("confidence", 0.0) or 0.0) for row in comparable}, reverse=True)
    thresholds = [max(thresholds) + 1e-12] + thresholds
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        coverage_values: list[float] = []
        success_values: list[float] = []
        for row in comparable:
            effective_abstain = bool(row.get("abstained")) or float(row.get("confidence", 0.0) or 0.0) < threshold
            correct = bool(row.get("correct"))
            action_target = row.get("derived_action_target")
            success = _revised_task_success(action_target, abstained=effective_abstain, correct=correct)
            if success is None:
                continue
            coverage_values.append(0.0 if effective_abstain else 1.0)
            success_values.append(1.0 if success else 0.0)
        if not success_values:
            continue
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(np.mean(coverage_values)),
                "risk": float(1.0 - np.mean(success_values)),
            }
        )
    return curve


def _mean_bool(rows_in: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows_in if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean([1.0 if value else 0.0 for value in values]))


def _group_metric(rows: list[dict[str, Any]], group_key: str, metric_key: str) -> dict[str, float]:
    groups = sorted({str(row.get(group_key) or "") for row in rows if row.get(group_key)})
    output: dict[str, float] = {}
    for group in groups:
        values = [row.get(metric_key) for row in rows if str(row.get(group_key) or "") == group and row.get(metric_key) is not None]
        if values:
            output[group] = float(np.mean([1.0 if value else 0.0 for value in values]))
    return output


def summarize_experimental_metrics(rows: list[dict[str, Any]], predictor_name: str) -> dict[str, Any]:
    coverage_values = [0.0 if row["abstained"] else 1.0 for row in rows]
    answered = [row for row in rows if not row["abstained"]]
    revised = [row for row in rows if row.get("task_success_revised") is not None]
    action_rows = [row for row in rows if row.get("action_accuracy_case") is not None]
    relation_rows = [row for row in rows if row.get("relation_accuracy_case") is not None]
    vision_info_rows = [row for row in rows if row.get("vision_info_accuracy_case") is not None]
    text_info_rows = [row for row in rows if row.get("text_info_accuracy_case") is not None]
    joint_info_rows = [row for row in rows if row.get("joint_info_accuracy_case") is not None]
    contradiction_rows = [row for row in rows if row.get("derived_pairwise_relation") == "contradictory"]
    irrelevance_rows = [
        row
        for row in rows
        if row.get("derived_pairwise_relation") in {"asymmetric", "both_weak"}
    ]
    contradiction_rows_for_targets = [row for row in rows if row.get("protocol_category") == "C4"]
    contradiction_answered = [row for row in contradiction_rows_for_targets if not row.get("abstained")]

    metrics = {
        "predictor_name": predictor_name,
        "examples": len(rows),
        "coverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
        "answer_accuracy": float(np.mean([1.0 if row["correct"] else 0.0 for row in rows])) if rows else 0.0,
        "accuracy_on_answered": float(np.mean([1.0 if row["correct"] else 0.0 for row in answered])) if answered else 0.0,
        "selective_accuracy": float(np.mean([1.0 if row["correct"] else 0.0 for row in answered])) if answered else 0.0,
        "task_success_revised": _mean_bool(revised, "task_success_revised"),
        "task_success_revised_count": len(revised),
        "task_success_legacy": _mean_bool(rows, "task_success_legacy"),
        "action_accuracy": _mean_bool(action_rows, "action_accuracy_case"),
        "action_count": len(action_rows),
        "vision_info_accuracy": _mean_bool(vision_info_rows, "vision_info_accuracy_case"),
        "vision_info_count": len(vision_info_rows),
        "text_info_accuracy": _mean_bool(text_info_rows, "text_info_accuracy_case"),
        "text_info_count": len(text_info_rows),
        "joint_info_accuracy": _mean_bool(joint_info_rows, "joint_info_accuracy_case"),
        "joint_info_count": len(joint_info_rows),
        "relation_accuracy": _mean_bool(relation_rows, "relation_accuracy_case"),
        "relation_count": len(relation_rows),
        "contradiction_error_rate": (1.0 - float(np.mean([1.0 if row["correct"] else 0.0 for row in contradiction_rows]))) if contradiction_rows else None,
        "irrelevance_error_rate": (1.0 - float(np.mean([1.0 if row["correct"] else 0.0 for row in irrelevance_rows]))) if irrelevance_rows else None,
        "legacy_semantics_mismatch_rate": _mean_bool(rows, "metric_semantics_mismatch"),
        "risk_coverage_task_success_revised": _confidence_threshold_curve(rows),
        "answer_accuracy_per_relation": {
            relation: float(np.mean([1.0 if row["correct"] else 0.0 for row in rows if row.get("derived_pairwise_relation") == relation]))
            for relation in sorted({str(row.get("derived_pairwise_relation")) for row in rows if row.get("derived_pairwise_relation")})
        },
        "task_success_per_category": _group_metric(rows, "protocol_category", "task_success_revised"),
        "answer_accuracy_per_category": _group_metric(rows, "protocol_category", "correct"),
        "abstain_rate_per_category": {
            category: float(np.mean([1.0 if row.get("abstained") else 0.0 for row in rows if row.get("protocol_category") == category]))
            for category in sorted({str(row.get("protocol_category")) for row in rows if row.get("protocol_category")})
        },
        "contradiction_vision_only_accuracy": _mean_bool(contradiction_rows_for_targets, "vision_matches_vision_target"),
        "contradiction_text_only_accuracy": _mean_bool(contradiction_rows_for_targets, "text_matches_text_target"),
        "contradiction_multimodal_abstain_rate": (
            float(np.mean([1.0 if row.get("abstained") else 0.0 for row in contradiction_rows_for_targets]))
            if contradiction_rows_for_targets
            else None
        ),
        "contradiction_multimodal_non_abstain_error_rate": (
            1.0 - float(np.mean([1.0 if row.get("correct") else 0.0 for row in contradiction_answered]))
        ) if contradiction_answered else None,
        "c2_vision_only_accuracy": _mean_bool(contradiction_rows_for_targets, "vision_matches_vision_target"),
        "c2_text_only_accuracy": _mean_bool(contradiction_rows_for_targets, "text_matches_text_target"),
        "c2_multimodal_abstain_rate": (
            float(np.mean([1.0 if row.get("abstained") else 0.0 for row in contradiction_rows_for_targets]))
            if contradiction_rows_for_targets
            else None
        ),
        "c2_multimodal_non_abstain_error_rate": (
            1.0 - float(np.mean([1.0 if row.get("correct") else 0.0 for row in contradiction_answered]))
        ) if contradiction_answered else None,
    }
    return metrics


class StructuredCARMPredictor:
    name = "structured_carm_experimental"

    def __init__(self, model: ExperimentalCARMHeads | CascadeCARMHeads | DistributionCARMHeads, backbone: BackboneAdapter, *, device: str = "cpu") -> None:
        self.model = model.to(torch.device(device))
        self.backbone = backbone
        self.device = torch.device(device)
        self._cache: dict[str, PredictionOutput] = {}

    @staticmethod
    def _vision_payload(example: ConflictExample) -> str:
        recipe = example.metadata.get("vision_recipe") if isinstance(example.metadata, dict) else None
        if isinstance(recipe, dict) and "payload" in recipe:
            return str(recipe["payload"])
        return example.image_path

    def predict(self, example: ConflictExample) -> PredictionOutput:
        if example.example_id in self._cache:
            cached = self._cache[example.example_id]
            return PredictionOutput(
                final_answer=cached.final_answer,
                abstained=cached.abstained,
                confidence=cached.confidence,
                metadata=dict(cached.metadata or {}),
                raw_text=cached.raw_text,
            )

        self.model.eval()
        with torch.no_grad():
            vision_payload = self._vision_payload(example)
            multimodal = self.backbone.run_backbone_multimodal(vision_payload, example.text_input, example.question)
            vision_probe = self.backbone.run_probe_vision_only(vision_payload, example.question)
            text_probe = self.backbone.run_probe_text_only(example.text_input, example.question)
            cross_size = getattr(getattr(self.model, "config", None), "cross_modal_feature_size", 5)
            _extract_fn = extract_cross_modal_features_augmented if cross_size >= 6 else extract_cross_modal_features
            phi_cross = _extract_fn(
                vision_probe.answer_dist,
                text_probe.answer_dist,
                vision_probe.answer_text,
                text_probe.answer_text,
            ).to(self.device)
            if isinstance(self.model, DistributionCARMHeads):
                vision_info_logits, text_info_logits, relation_logits, action_logits = self.model(
                    multimodal.answer_dist.to(self.device),
                    vision_probe.answer_dist.to(self.device),
                    text_probe.answer_dist.to(self.device),
                    phi_cross,
                )
            else:
                vision_info_logits, text_info_logits, relation_logits, action_logits = self.model(
                    multimodal.hidden_states.to(self.device),
                    vision_probe.features.to(self.device),
                    text_probe.features.to(self.device),
                    phi_cross,
                )
            pred_vision_state = decode_info_state(vision_info_logits)
            pred_text_state = decode_info_state(text_info_logits)
            pred_relation = decode_pairwise_relation(relation_logits)
            pred_action = decode_action(action_logits)
            pred_joint_state = joint_info_state_label(pred_vision_state, pred_text_state)
            final_answer, abstained, audit = apply_action_and_generate(pred_action, vision_probe, text_probe, family=example.family)
            prediction = PredictionOutput(
                final_answer=final_answer,
                abstained=bool(abstained),
                confidence=float(torch.softmax(action_logits, dim=-1).max().item()),
                raw_text=multimodal.raw_text,
                metadata={
                    "pred_action": pred_action.value,
                    "pred_joint_info_state": pred_joint_state,
                    "pred_vision_info_state": pred_vision_state,
                    "pred_text_info_state": pred_text_state,
                    "pred_pairwise_relation": pred_relation,
                    "audit": audit,
                    "multimodal_raw_output": multimodal.raw_text,
                    "vision_raw_output": vision_probe.raw_text,
                    "text_raw_output": text_probe.raw_text,
                    **_result_metadata(multimodal),
                    **{
                        key: value
                        for key, value in _result_metadata(vision_probe).items()
                        if key not in {"projection_succeeded", "used_fallback_dist"}
                    },
                },
            )
        self._cache[example.example_id] = prediction
        return PredictionOutput(
            final_answer=prediction.final_answer,
            abstained=prediction.abstained,
            confidence=prediction.confidence,
            metadata=dict(prediction.metadata or {}),
            raw_text=prediction.raw_text,
        )


def collect_predictor_diagnostics(
    predictor: Any,
    backbone: BackboneAdapter,
    example: ConflictExample,
) -> PredictorDiagnostics:
    prediction = _as_prediction_output(predictor.predict(example))
    vision_payload = example.image_path
    recipe = example.metadata.get("vision_recipe") if isinstance(example.metadata, dict) else None
    if isinstance(recipe, dict) and "payload" in recipe:
        vision_payload = str(recipe["payload"])

    multimodal = backbone.run_backbone_multimodal(vision_payload, example.text_input, example.question)
    vision_probe = backbone.run_probe_vision_only(vision_payload, example.question)
    text_probe = backbone.run_probe_text_only(example.text_input, example.question)

    return PredictorDiagnostics(
        prediction=prediction,
        multimodal_answer=multimodal.answer_text,
        multimodal_confidence=_prediction_confidence(multimodal.answer_dist),
        multimodal_entropy=_dist_entropy(multimodal.answer_dist),
        multimodal_margin=_dist_margin(multimodal.answer_dist),
        vision_answer=vision_probe.answer_text,
        vision_confidence=_prediction_confidence(vision_probe.answer_dist),
        vision_entropy=_dist_entropy(vision_probe.answer_dist),
        vision_margin=_dist_margin(vision_probe.answer_dist),
        text_answer=text_probe.answer_text,
        text_confidence=_prediction_confidence(text_probe.answer_dist),
        text_entropy=_dist_entropy(text_probe.answer_dist),
        text_margin=_dist_margin(text_probe.answer_dist),
        multimodal_raw_output=multimodal.raw_text,
        vision_raw_output=vision_probe.raw_text,
        text_raw_output=text_probe.raw_text,
    )


def evaluate_predictor_experimental(
    predictor: Any,
    backbone: BackboneAdapter,
    examples: list[ConflictExample],
    derived_by_id: dict[str, DerivedLabels],
    *,
    output_dir: str | Path,
    canonicalization_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canon_cfg = CanonicalizationConfig.from_mapping(canonicalization_cfg or {})

    rows: list[dict[str, Any]] = []
    for example in examples:
        derived = derived_by_id[example.example_id]
        diagnostics = collect_predictor_diagnostics(predictor, backbone, example)
        prediction = diagnostics.prediction
        metadata = dict(prediction.metadata or {})
        pred_vision_info_state = metadata.get("pred_vision_info_state")
        pred_text_info_state = metadata.get("pred_text_info_state")
        pred_joint_info_state = metadata.get("pred_joint_info_state")
        pred_pairwise_relation = metadata.get("pred_pairwise_relation")
        pred_action = metadata.get("pred_action")
        answered = not bool(prediction.abstained)
        correct = _answers_match(prediction.final_answer, example.gold_answer, example, canon_cfg)
        legacy_task_success = task_success_from_components(
            example.oracle_action.value,
            pred_action,
            prediction.abstained,
            correct,
            protocol_category=_effective_protocol_category(example),
        )
        revised_task_success = _revised_task_success(
            derived.action_target,
            abstained=bool(prediction.abstained),
            correct=bool(correct),
        )
        metric_mismatch = False
        if revised_task_success is not None:
            metric_mismatch = bool(legacy_task_success) != bool(revised_task_success)

        vision_matches_target = (
            _answers_match(diagnostics.vision_answer, example.vision_supported_target or "", example, canon_cfg)
            if example.vision_supported_target
            else None
        )
        text_matches_target = (
            _answers_match(diagnostics.text_answer, example.text_supported_target or "", example, canon_cfg)
            if example.text_supported_target
            else None
        )

        row = {
            "example_id": example.example_id,
            "split": example.split.value,
            "protocol_category": _effective_protocol_category(example),
            "family": example.family.value,
            "question": example.question,
            "gold_answer": example.gold_answer,
            "gold_action_legacy": example.oracle_action.value,
            "derived_action_target": derived.action_target,
            "derived_action_target_available": derived.action_target_available,
            "derived_vision_info_state": derived.vision_info_state,
            "derived_text_info_state": derived.text_info_state,
            "derived_joint_info_state": derived.joint_info_state,
            "derived_pairwise_relation": derived.pairwise_relation,
            "derived_reason": derived.reason,
            "derivation_status": derived.derivation_status,
            "vision_supported_target": example.vision_supported_target,
            "text_supported_target": example.text_supported_target,
            "multimodal_answer": diagnostics.multimodal_answer,
            "vision_answer": diagnostics.vision_answer,
            "text_answer": diagnostics.text_answer,
            "vision_matches_vision_target": vision_matches_target,
            "text_matches_text_target": text_matches_target,
            "multimodal_confidence": diagnostics.multimodal_confidence,
            "vision_confidence": diagnostics.vision_confidence,
            "text_confidence": diagnostics.text_confidence,
            "multimodal_entropy": diagnostics.multimodal_entropy,
            "vision_entropy": diagnostics.vision_entropy,
            "text_entropy": diagnostics.text_entropy,
            "multimodal_margin": diagnostics.multimodal_margin,
            "vision_margin": diagnostics.vision_margin,
            "text_margin": diagnostics.text_margin,
            "pred_vision_info_state": pred_vision_info_state,
            "pred_text_info_state": pred_text_info_state,
            "pred_joint_info_state": pred_joint_info_state,
            "pred_pairwise_relation": pred_pairwise_relation,
            "pred_action": pred_action,
            "final_answer": prediction.final_answer,
            "abstained": bool(prediction.abstained),
            "answered": bool(answered),
            "correct": bool(correct),
            "confidence": float(prediction.confidence or 0.0),
            "task_success_legacy": bool(legacy_task_success),
            "task_success_revised": revised_task_success,
            "metric_semantics_mismatch": bool(metric_mismatch),
            "vision_info_accuracy_case": (
                pred_vision_info_state == derived.vision_info_state
                if pred_vision_info_state is not None and derived.vision_info_state is not None
                else None
            ),
            "text_info_accuracy_case": (
                pred_text_info_state == derived.text_info_state
                if pred_text_info_state is not None and derived.text_info_state is not None
                else None
            ),
            "joint_info_accuracy_case": (
                pred_vision_info_state == derived.vision_info_state and pred_text_info_state == derived.text_info_state
                if pred_vision_info_state is not None
                and pred_text_info_state is not None
                and derived.vision_info_state is not None
                and derived.text_info_state is not None
                else None
            ),
            "relation_accuracy_case": (
                pred_pairwise_relation == derived.pairwise_relation
                if pred_pairwise_relation is not None and derived.pairwise_relation is not None
                else None
            ),
            "action_accuracy_case": (
                pred_action == derived.action_target
                if pred_action is not None and derived.action_target_available and derived.action_target is not None
                else None
            ),
            "c2_vision_only_correct": vision_matches_target if _effective_protocol_category(example) == "C4" else None,
            "c2_text_only_correct": text_matches_target if _effective_protocol_category(example) == "C4" else None,
            "c2_multimodal_abstained": bool(prediction.abstained) if _effective_protocol_category(example) == "C4" else None,
            "multimodal_raw_output": diagnostics.multimodal_raw_output,
            "vision_raw_output": diagnostics.vision_raw_output,
            "text_raw_output": diagnostics.text_raw_output,
            "raw_output": prediction.raw_text,
            "audit": metadata.get("audit"),
            "failure_reason": None,
        }

        for key in (*DIAGNOSTIC_BOOL_KEYS, *DIAGNOSTIC_VALUE_KEYS):
            if key in metadata:
                row[key] = metadata[key]

        failure_reasons: list[str] = []
        if revised_task_success is False:
            failure_reasons.append("revised_task_failure")
        if not correct and not prediction.abstained:
            failure_reasons.append("answered_incorrectly")
        if derived.action_target_available and pred_action is not None and pred_action != derived.action_target:
            failure_reasons.append("action_mismatch")
        if pred_pairwise_relation is not None and derived.pairwise_relation is not None and pred_pairwise_relation != derived.pairwise_relation:
            failure_reasons.append("relation_mismatch")
        if pred_vision_info_state is not None and derived.vision_info_state is not None and pred_vision_info_state != derived.vision_info_state:
            failure_reasons.append("vision_info_mismatch")
        if pred_text_info_state is not None and derived.text_info_state is not None and pred_text_info_state != derived.text_info_state:
            failure_reasons.append("text_info_mismatch")
        if metric_mismatch:
            failure_reasons.append("legacy_vs_revised_metric_mismatch")
        if failure_reasons:
            row["failure_reason"] = ";".join(failure_reasons)

        rows.append(row)

    metrics = summarize_experimental_metrics(rows, str(getattr(predictor, "name", type(predictor).__name__)))
    predictions_jsonl = out_dir / "per_example_predictions.jsonl"
    predictions_csv = out_dir / "per_example_predictions.csv"
    metrics_path = out_dir / "metrics.json"

    with predictions_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_failure_diagnostics(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    failures: list[dict[str, Any]] = []
    for row in rows:
        if row.get("failure_reason") is None and not row.get("metric_semantics_mismatch"):
            continue

        failure_type = "other_failure"
        if row.get("protocol_category") == "C4" and row.get("abstained") is False:
            failure_type = "c4_contradiction_collapse"
        elif row.get("pred_pairwise_relation") == "contradictory" and row.get("derived_pairwise_relation") != "contradictory":
            failure_type = "false_contradiction"
        elif row.get("vision_info_accuracy_case") is False or row.get("text_info_accuracy_case") is False:
            failure_type = "informativeness_error"
        elif row.get("derived_pairwise_relation") == "asymmetric" and row.get("abstained") is True:
            failure_type = "wrong_abstention"
        elif row.get("correct") is False and float(row.get("confidence", 0.0) or 0.0) >= 0.75 and row.get("abstained") is False:
            failure_type = "overconfident_wrong_answer"

        failures.append(
            {
                "example_id": row.get("example_id"),
                "protocol_category": row.get("protocol_category"),
                "failure_type": failure_type,
                "failure_reason": row.get("failure_reason"),
                "vision_answer": row.get("vision_answer"),
                "text_answer": row.get("text_answer"),
                "multimodal_answer": row.get("multimodal_answer"),
                "vision_supported_target": row.get("vision_supported_target"),
                "text_supported_target": row.get("text_supported_target"),
                "unimodal_answers_agree": row.get("vision_answer") == row.get("text_answer"),
                "multimodal_confidence": row.get("multimodal_confidence"),
                "vision_entropy": row.get("vision_entropy"),
                "text_entropy": row.get("text_entropy"),
                "confidence_gap": abs(float(row.get("vision_confidence", 0.0) or 0.0) - float(row.get("text_confidence", 0.0) or 0.0)),
                "derived_vision_info_state": row.get("derived_vision_info_state"),
                "pred_vision_info_state": row.get("pred_vision_info_state"),
                "derived_text_info_state": row.get("derived_text_info_state"),
                "pred_text_info_state": row.get("pred_text_info_state"),
                "derived_pairwise_relation": row.get("derived_pairwise_relation"),
                "pred_pairwise_relation": row.get("pred_pairwise_relation"),
                "derived_action_target": row.get("derived_action_target"),
                "pred_action": row.get("pred_action"),
                "abstained": row.get("abstained"),
                "correct": row.get("correct"),
                "task_success_revised": row.get("task_success_revised"),
                "metric_semantics_mismatch": row.get("metric_semantics_mismatch"),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in failures for key in row.keys()}) if failures else [
        "example_id",
        "failure_type",
        "failure_reason",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def summarize_feature_diagnostics(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_vision_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_text_state: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        relation = str(row.get("derived_pairwise_relation") or "")
        if relation:
            by_relation[relation].append(row)
        vision_state = str(row.get("derived_vision_info_state") or "")
        if vision_state:
            by_vision_state[vision_state].append(row)
        text_state = str(row.get("derived_text_info_state") or "")
        if text_state:
            by_text_state[text_state].append(row)

    lines = [
        "# Feature Diagnostics Summary",
        "",
        "This summary is descriptive. It is meant to show whether the current feature bundle moves with the revised supervision targets.",
        "",
    ]

    def _append_group_block(title: str, groups: dict[str, list[dict[str, Any]]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for feature_key in ("multimodal_entropy", "vision_entropy", "text_entropy", "multimodal_margin", "vision_margin", "text_margin"):
            lines.append(f"### {feature_key}")
            for label in sorted(groups):
                values = [float(row.get(feature_key, 0.0) or 0.0) for row in groups[label]]
                if values:
                    lines.append(f"- {label}: mean={float(np.mean(values)):.4f}, std={float(np.std(values)):.4f}, n={len(values)}")
            lines.append("")

    _append_group_block("By Pairwise Relation", by_relation)
    _append_group_block("By Vision Informativeness", by_vision_state)
    _append_group_block("By Text Informativeness", by_text_state)

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
