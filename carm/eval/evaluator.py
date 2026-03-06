from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from carm.data.io import read_jsonl
from carm.data.schema import Family
from carm.data.schema import ConflictExample
from carm.eval.canonicalization import CanonicalizationConfig, canonicalize_answer
from carm.eval.metrics import summarize_metrics, task_success_from_components
from carm.eval.types import AnswerOutput, PolicyOutput, PredictionOutput
from carm.models.carm_model import CARMHeads, select_action
from carm.models.interfaces import BackboneAdapter
from carm.models.policy import apply_action_and_generate


PREDICTIONS_FILENAME = "per_example_predictions.jsonl"
METRICS_FILENAME = "metrics.json"
ABSTAIN_ANSWER = "<ABSTAIN>"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{max(0.0, seconds):.1f}s"

    total_seconds = max(0, int(seconds))
    mins, secs = divmod(total_seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:d}h{mins:02d}m{secs:02d}s"
    return f"{mins:d}m{secs:02d}s"


class CARMPredictor:
    name = "carm"

    def __init__(self, model: CARMHeads, backbone: BackboneAdapter, device: str = "cpu") -> None:
        self.model = model.to(torch.device(device))
        self.backbone = backbone
        self.device = torch.device(device)
        self._cache: dict[str, PredictionOutput] = {}

    @staticmethod
    def _vision_payload(ex: ConflictExample) -> str:
        recipe = ex.metadata.get("vision_recipe") if isinstance(ex.metadata, dict) else None
        if isinstance(recipe, dict) and "payload" in recipe:
            return str(recipe["payload"])
        return ex.image_path

    def _infer_once(self, ex: ConflictExample) -> PredictionOutput:
        key = ex.example_id
        if key in self._cache:
            return self._cache[key]

        self.model.eval()
        with torch.no_grad():
            mm = self.backbone.run_backbone_multimodal(self._vision_payload(ex), ex.text_input, ex.question)
            pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
            pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

            conflict_logits, reliability, action_logits = self.model.carm_forward(
                anchor_states=mm.hidden_states.to(self.device),
                phi_v=pv.features.to(self.device),
                phi_t=pt.features.to(self.device),
            )
            action = select_action(action_logits)
            final_answer, abstained, audit = apply_action_and_generate(action, pv, pt)
            if abstained:
                final_answer = ABSTAIN_ANSWER

            pred_conf_idx = int(torch.argmax(conflict_logits, dim=-1).item())
            conflict_map = [
                Family.NONE.value,
                Family.EXISTENCE.value,
                Family.COUNT.value,
                Family.ATTRIBUTE_COLOR.value,
            ]
            pred_conflict_type = conflict_map[min(pred_conf_idx, len(conflict_map) - 1)]
            reliability_vec = reliability.squeeze(0)

            pred = PredictionOutput(
                final_answer=final_answer,
                abstained=bool(abstained),
                confidence=float(torch.softmax(action_logits, dim=-1).max().item()),
                metadata={
                    "pred_action": action.value,
                    "pred_conflict_type": pred_conflict_type,
                    "r_v": float(reliability_vec[0].item()),
                    "r_t": float(reliability_vec[1].item()),
                    "audit": audit,
                },
            )

        self._cache[key] = pred
        return pred

    def predict(self, ex: ConflictExample) -> PredictionOutput:
        return self._infer_once(ex)


def _row_example_id(row: dict[str, Any]) -> str:
    if "example_id" in row:
        return str(row.get("example_id", ""))
    example = row.get("example")
    if isinstance(example, dict):
        return str(example.get("example_id", ""))
    return ""


def _row_is_compatible(row: dict[str, Any]) -> bool:
    required = {
        "example_id",
        "base_id",
        "image_path",
        "text_input",
        "question",
        "gold_answer",
        "split",
        "family",
        "oracle_action",
        "protocol_category",
        "final_answer",
        "abstained",
        "confidence",
        "correct",
        "task_success",
    }
    return required.issubset(set(row.keys()))


def _is_correct_answer(pred_text: str, gold_text: str, ex: ConflictExample, cfg: CanonicalizationConfig) -> bool:
    pred = canonicalize_answer(pred_text, ex.answer_type, cfg=cfg)
    gold = canonicalize_answer(gold_text, ex.answer_type, cfg=cfg)

    if pred.canonical_label is not None and gold.canonical_label is not None:
        return pred.canonical_label == gold.canonical_label

    return bool(pred.normalized_text) and pred.normalized_text == gold.normalized_text


def _protocol_category(ex: ConflictExample) -> str:
    if isinstance(ex.metadata, dict):
        return str(ex.metadata.get("protocol_category", "")).strip()
    return ""


def _coerce_prediction(raw: Any) -> PredictionOutput:
    if isinstance(raw, PredictionOutput):
        return PredictionOutput(
            final_answer=str(raw.final_answer),
            abstained=bool(raw.abstained),
            confidence=float(raw.confidence or 0.0),
            metadata=raw.metadata or {},
        )

    if isinstance(raw, dict):
        return PredictionOutput(
            final_answer=str(raw.get("final_answer", "")),
            abstained=bool(raw.get("abstained", False)),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
        )

    final_answer = getattr(raw, "final_answer", None)
    abstained = getattr(raw, "abstained", None)
    confidence = getattr(raw, "confidence", None)
    if final_answer is None or abstained is None:
        raise TypeError("Predictor output must expose final_answer, abstained, and confidence.")
    metadata = getattr(raw, "metadata", None)
    return PredictionOutput(
        final_answer=str(final_answer),
        abstained=bool(abstained),
        confidence=float(confidence or 0.0),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _predict_flat(predictor: Any, ex: ConflictExample) -> PredictionOutput:
    if hasattr(predictor, "predict") and callable(getattr(predictor, "predict")):
        return _coerce_prediction(predictor.predict(ex))

    if hasattr(predictor, "predict_answer") and callable(getattr(predictor, "predict_answer")):
        answer: AnswerOutput = predictor.predict_answer(ex)
        policy: PolicyOutput | None = None
        if hasattr(predictor, "predict_policy") and callable(getattr(predictor, "predict_policy")):
            policy = predictor.predict_policy(ex)

        final_answer = str(answer.raw_text)
        abstained = False
        confidence = float(answer.answer_confidence or 0.0)
        metadata = dict(answer.metadata or {})

        if policy is not None:
            abstained = bool(policy.abstained)
            confidence = float(policy.policy_confidence or answer.answer_confidence or 0.0)
            metadata["pred_action"] = str(policy.pred_action)
            metadata["pred_conflict_type"] = str(policy.pred_conflict_type)
            if policy.r_v is not None:
                metadata["r_v"] = float(policy.r_v)
            if policy.r_t is not None:
                metadata["r_t"] = float(policy.r_t)
            if policy.audit is not None:
                metadata["audit"] = policy.audit
            if abstained:
                final_answer = ABSTAIN_ANSWER

        return PredictionOutput(
            final_answer=final_answer,
            abstained=abstained,
            confidence=confidence,
            metadata=metadata,
        )

    raise TypeError(f"Predictor {type(predictor).__name__} must define predict() or predict_answer().")


def evaluate_predictor(
    predictor: Any,
    examples: list[ConflictExample],
    output_dir: str | Path,
    *,
    track: str = "all",
    schema_version: str = "2.0",
    resume: bool = False,
    progress_every: int = 500,
    log_fn: Callable[[str], None] | None = None,
    semantic_match_threshold: float = 0.82,
    canonicalization_cfg: CanonicalizationConfig | dict[str, Any] | None = None,
    include_heuristic_calibration: bool = False,
) -> dict[str, Any]:
    del schema_version
    del semantic_match_threshold
    del include_heuristic_calibration

    if track not in {"answer", "policy", "all"}:
        raise ValueError(f"Unsupported track: {track}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canon_cfg = (
        canonicalization_cfg
        if isinstance(canonicalization_cfg, CanonicalizationConfig)
        else CanonicalizationConfig.from_mapping(canonicalization_cfg)
    )

    total = len(examples)
    target_ids = {ex.example_id for ex in examples}
    predictions_path = out_dir / PREDICTIONS_FILENAME

    records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    def _emit(msg: str) -> None:
        if log_fn is None:
            print(msg)
        else:
            log_fn(msg)

    if resume and predictions_path.exists():
        existing = read_jsonl(predictions_path)
        restart_required = False
        for row in existing:
            ex_id = _row_example_id(row)
            if not ex_id or ex_id not in target_ids or ex_id in completed_ids:
                continue
            if not _row_is_compatible(row):
                restart_required = True
                break
            records.append(row)
            completed_ids.add(ex_id)

        if restart_required:
            _emit(f"[{predictor.name}] resume data incompatible with flattened evaluator schema; restarting fresh outputs")
            records = []
            completed_ids = set()
        elif completed_ids:
            _emit(f"[{predictor.name}] resume loaded {len(completed_ids)}/{total} predictions from {predictions_path}")

    mode = "a" if resume and predictions_path.exists() and completed_ids else "w"
    processed = len(completed_ids)
    resumed_count = processed
    start = time.time()

    with predictions_path.open(mode, encoding="utf-8") as f:
        for ex in examples:
            if ex.example_id in completed_ids:
                continue

            prediction = _predict_flat(predictor, ex)
            if prediction.abstained:
                prediction.final_answer = ABSTAIN_ANSWER

            correct = _is_correct_answer(prediction.final_answer, ex.gold_answer, ex, canon_cfg)
            metadata = prediction.metadata or {}
            task_success = task_success_from_components(
                ex.oracle_action.value,
                metadata.get("pred_action"),
                prediction.abstained,
                correct,
                protocol_category=_protocol_category(ex),
            )

            row = {
                "example_id": ex.example_id,
                "base_id": ex.base_id,
                "image_path": ex.image_path,
                "text_input": ex.text_input,
                "question": ex.question,
                "gold_answer": ex.gold_answer,
                "split": ex.split.value,
                "family": ex.family.value,
                "oracle_action": ex.oracle_action.value,
                "protocol_category": _protocol_category(ex),
                "final_answer": prediction.final_answer,
                "abstained": bool(prediction.abstained),
                "confidence": float(prediction.confidence or 0.0),
                "correct": bool(correct),
                "task_success": bool(task_success),
            }
            for key in ("pred_action", "pred_conflict_type", "r_v", "r_t", "audit"):
                if key in metadata:
                    row[key] = metadata[key]

            records.append(row)
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()

            processed += 1
            if progress_every > 0 and (processed % progress_every == 0 or processed == total):
                elapsed = max(1e-6, time.time() - start)
                completed_now = processed - resumed_count
                rate = completed_now / elapsed if completed_now > 0 else 0.0
                pct = (processed / max(1, total)) * 100.0
                remaining = max(0, total - processed)
                eta_text = _format_duration(remaining / rate) if rate > 0 else "n/a"
                _emit(
                    f"[{predictor.name}] progress {processed}/{total} ({pct:.1f}%) "
                    f"elapsed={_format_duration(elapsed)} eta={eta_text} rate={rate:.2f} ex/s"
                )

    metrics = summarize_metrics(records)
    with (out_dir / METRICS_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
