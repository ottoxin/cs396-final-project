from __future__ import annotations

import json
import time
from hashlib import sha256
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
RUN_METADATA_FILENAME = "run_metadata.json"
ABSTAIN_ANSWER = "<ABSTAIN>"
RAW_METADATA_KEYS = (
    "multimodal_raw_output",
    "vision_raw_output",
    "text_raw_output",
)
DIAGNOSTIC_BOOL_KEYS = (
    "projection_succeeded",
    "used_fallback_dist",
    "parsed_unknown",
    "parsed_in_active_vocab",
    "out_of_vocab_generation",
    "parsed_argmax_agree",
)
DIAGNOSTIC_VALUE_KEYS = (
    "canonicalized_candidate",
    "dist_argmax_label",
)
C2_DIAGNOSTIC_KEYS = (
    "c2_vision_only_correct",
    "c2_text_only_correct",
    "c2_multimodal_abstained",
)


def _result_metadata(result: Any) -> dict[str, Any]:
    return dict(getattr(result, "metadata", None) or {})


def _merge_probe_metadata(vision_result: Any, text_result: Any) -> dict[str, Any]:
    vision_meta = _result_metadata(vision_result)
    text_meta = _result_metadata(text_result)
    merged: dict[str, Any] = {
        "vision_raw_output": getattr(vision_result, "raw_text", None),
        "text_raw_output": getattr(text_result, "raw_text", None),
    }
    for key in DIAGNOSTIC_BOOL_KEYS:
        v_val = bool(vision_meta.get(key, False))
        t_val = bool(text_meta.get(key, False))
        if key in {"projection_succeeded", "parsed_in_active_vocab", "parsed_argmax_agree"}:
            merged[key] = v_val and t_val
        else:
            merged[key] = v_val or t_val
    for key in DIAGNOSTIC_VALUE_KEYS:
        v_val = vision_meta.get(key)
        t_val = text_meta.get(key)
        merged[key] = v_val if v_val == t_val else None
    return merged


def _vision_payload(ex: ConflictExample) -> str:
    recipe = ex.metadata.get("vision_recipe") if isinstance(ex.metadata, dict) else None
    if isinstance(recipe, dict) and "payload" in recipe:
        return str(recipe["payload"])
    return ex.image_path


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

    def __init__(
        self,
        model: CARMHeads,
        backbone: BackboneAdapter,
        device: str = "cpu",
        diagnostic_validity: dict[str, bool] | None = None,
    ) -> None:
        self.model = model.to(torch.device(device))
        self.backbone = backbone
        self.device = torch.device(device)
        validity = diagnostic_validity or {}
        self.diagnostic_validity = {
            "conflict": bool(validity.get("conflict", True)),
            "reliability": bool(validity.get("reliability", True)),
        }
        self._cache: dict[str, PredictionOutput] = {}

    @staticmethod
    def _vision_payload(ex: ConflictExample) -> str:
        return _vision_payload(ex)

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
            final_answer, abstained, audit = apply_action_and_generate(
                action,
                pv,
                pt,
                family=ex.family,
            )
            if abstained:
                final_answer = ABSTAIN_ANSWER
            pv_meta = _result_metadata(pv)
            pt_meta = _result_metadata(pt)
            merged_probe_meta = _merge_probe_metadata(pv, pt)
            raw_text = None
            audit_path = str(audit.get("path", ""))
            prediction_meta: dict[str, Any]
            if audit_path in {"vision_only", "require_agreement"}:
                raw_text = pv.raw_text
                prediction_meta = merged_probe_meta if audit_path == "require_agreement" else dict(pv_meta)
            elif audit_path == "text_only":
                raw_text = pt.raw_text
                prediction_meta = dict(pt_meta)
            else:
                prediction_meta = merged_probe_meta

            pred_conflict_type: str | None = None
            if self.diagnostic_validity["conflict"]:
                pred_conf_idx = int(torch.argmax(conflict_logits, dim=-1).item())
                conflict_map = [
                    Family.NONE.value,
                    Family.EXISTENCE.value,
                    Family.COUNT.value,
                    Family.ATTRIBUTE_COLOR.value,
                ]
                pred_conflict_type = conflict_map[min(pred_conf_idx, len(conflict_map) - 1)]
            reliability_vec = reliability.squeeze(0)
            r_v = float(reliability_vec[0].item()) if self.diagnostic_validity["reliability"] else None
            r_t = float(reliability_vec[1].item()) if self.diagnostic_validity["reliability"] else None

            pred = PredictionOutput(
                final_answer=final_answer,
                abstained=bool(abstained),
                confidence=float(torch.softmax(action_logits, dim=-1).max().item()),
                raw_text=raw_text,
                metadata={
                    **prediction_meta,
                    "pred_action": action.value,
                    "pred_conflict_type": pred_conflict_type,
                    "multimodal_raw_output": mm.raw_text,
                    "vision_raw_output": pv.raw_text,
                    "text_raw_output": pt.raw_text,
                    "r_v": r_v,
                    "r_t": r_t,
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
        "raw_output",
        "correct",
        "task_success",
        *DIAGNOSTIC_BOOL_KEYS,
        *DIAGNOSTIC_VALUE_KEYS,
    }
    return required.issubset(set(row.keys()))


def _build_run_metadata(
    *,
    predictor_name: str,
    resolved_config_hash: str | None,
    selected_split: str | None,
    dataset_manifest_hash: str | None,
    git_commit: str | None,
    unsafe_resume_override: bool = False,
    override_reason: str | None = None,
) -> dict[str, Any]:
    components = {
        "resolved_config_hash": resolved_config_hash,
        "selected_split": selected_split,
        "predictor_name": predictor_name,
        "dataset_manifest_hash": dataset_manifest_hash,
        "git_commit": git_commit,
    }
    missing = [key for key, value in components.items() if not value]
    fingerprint = None
    if not missing:
        fingerprint = sha256(
            json.dumps(components, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return {
        **components,
        "fingerprint": fingerprint,
        "strict_resume_supported": not missing,
        "missing_components": missing,
        "unsafe_resume_override": bool(unsafe_resume_override),
        "override_reason": override_reason,
    }


def _load_run_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _write_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _fingerprint_mismatch_fields(expected: dict[str, Any], existing: dict[str, Any] | None) -> list[str]:
    if existing is None:
        return ["run_metadata"]

    mismatched: list[str] = []
    for key in (
        "resolved_config_hash",
        "selected_split",
        "predictor_name",
        "dataset_manifest_hash",
        "git_commit",
        "fingerprint",
    ):
        if existing.get(key) != expected.get(key):
            mismatched.append(key)
    return mismatched


def _is_correct_answer(pred_text: str, gold_text: str, ex: ConflictExample, cfg: CanonicalizationConfig) -> bool:
    return _answers_match(pred_text, gold_text, ex.answer_type, cfg)


def _answers_match(
    pred_text: str,
    gold_text: str,
    answer_type: Any,
    cfg: CanonicalizationConfig,
) -> bool:
    pred = canonicalize_answer(pred_text, answer_type, cfg=cfg)
    gold = canonicalize_answer(gold_text, answer_type, cfg=cfg)

    if pred.canonical_label is not None and gold.canonical_label is not None:
        return pred.canonical_label == gold.canonical_label

    return bool(pred.normalized_text) and pred.normalized_text == gold.normalized_text


def _protocol_category(ex: ConflictExample) -> str:
    if isinstance(ex.metadata, dict):
        return str(ex.metadata.get("protocol_category", "")).strip()
    return ""


def _effective_protocol_category(ex: ConflictExample) -> str:
    relation = str(ex.pairwise_relation or "").strip().lower()
    vision_info = str(ex.vision_info_state or "").strip().lower()
    text_info = str(ex.text_info_state or "").strip().lower()
    if relation == "consistent":
        return "C1"
    if relation == "contradictory":
        return "C4"
    if relation == "both_weak":
        return "C5"
    if relation == "asymmetric":
        if vision_info == "informative" and text_info == "uninformative":
            return "C2"
        if vision_info == "uninformative" and text_info == "informative":
            return "C3"

    category = _protocol_category(ex).upper()
    legacy_to_hf = {
        "C1": "C1",
        "C2": "C4",
        "C3": "C2",
        "C4": "C3",
        "C5": "C5",
    }
    return legacy_to_hf.get(category, category)


def _strip_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _c2_vision_supported_answer(ex: ConflictExample) -> str | None:
    if _effective_protocol_category(ex) != "C4":
        return None
    explicit = _strip_optional_text(ex.vision_supported_target)
    if explicit is not None:
        return explicit
    return _strip_optional_text(ex.gold_answer)


def _c2_text_supported_answer(ex: ConflictExample) -> str | None:
    explicit = _strip_optional_text(ex.text_supported_target)
    if explicit is not None:
        return explicit
    if not isinstance(ex.metadata, dict):
        return None
    return _strip_optional_text(ex.metadata.get("c2_text_supported_answer"))


def _probe_backbone(predictor: Any) -> BackboneAdapter | None:
    backbone = getattr(predictor, "backbone", None)
    if backbone is None:
        return None
    if not callable(getattr(backbone, "run_probe_vision_only", None)):
        return None
    if not callable(getattr(backbone, "run_probe_text_only", None)):
        return None
    return backbone


def _compute_c2_diagnostics(
    predictor: Any,
    ex: ConflictExample,
    prediction: PredictionOutput,
    cfg: CanonicalizationConfig,
) -> dict[str, bool | None]:
    diagnostics = {
        "c2_vision_only_correct": None,
        "c2_text_only_correct": None,
        "c2_multimodal_abstained": None,
    }
    if _effective_protocol_category(ex) != "C4":
        return diagnostics

    diagnostics["c2_multimodal_abstained"] = bool(prediction.abstained)

    backbone = _probe_backbone(predictor)
    if backbone is None:
        return diagnostics

    vision_supported_answer = _c2_vision_supported_answer(ex)
    if vision_supported_answer is not None:
        vision_probe = backbone.run_probe_vision_only(_vision_payload(ex), ex.question)
        diagnostics["c2_vision_only_correct"] = _answers_match(
            vision_probe.answer_text,
            vision_supported_answer,
            ex.answer_type,
            cfg,
        )

    text_supported_answer = _c2_text_supported_answer(ex)
    if text_supported_answer is not None:
        text_probe = backbone.run_probe_text_only(ex.text_input, ex.question)
        diagnostics["c2_text_only_correct"] = _answers_match(
            text_probe.answer_text,
            text_supported_answer,
            ex.answer_type,
            cfg,
        )
    return diagnostics


def _coerce_prediction(raw: Any) -> PredictionOutput:
    if isinstance(raw, PredictionOutput):
        return PredictionOutput(
            final_answer=str(raw.final_answer),
            abstained=bool(raw.abstained),
            confidence=float(raw.confidence or 0.0),
            metadata=raw.metadata or {},
            raw_text=getattr(raw, "raw_text", None),
        )

    if isinstance(raw, dict):
        return PredictionOutput(
            final_answer=str(raw.get("final_answer", "")),
            abstained=bool(raw.get("abstained", False)),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
            raw_text=str(raw["raw_text"]) if raw.get("raw_text") is not None else None,
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
        raw_text=getattr(raw, "raw_text", None),
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
        metadata.setdefault("raw_output", str(answer.raw_text))

        if policy is not None:
            abstained = bool(policy.abstained)
            confidence = float(policy.policy_confidence or answer.answer_confidence or 0.0)
            metadata["pred_action"] = str(policy.pred_action)
            metadata["pred_conflict_type"] = (
                str(policy.pred_conflict_type)
                if policy.pred_conflict_type is not None
                else None
            )
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
            raw_text=str(answer.raw_text),
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
    resolved_config_hash: str | None = None,
    selected_split: str | None = None,
    dataset_manifest_hash: str | None = None,
    git_commit: str | None = None,
    unsafe_resume_override: bool = False,
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
    metadata_path = out_dir / RUN_METADATA_FILENAME

    records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    def _emit(msg: str) -> None:
        if log_fn is None:
            print(msg)
        else:
            log_fn(msg)

    run_metadata = _build_run_metadata(
        predictor_name=str(getattr(predictor, "name", type(predictor).__name__)),
        resolved_config_hash=resolved_config_hash,
        selected_split=selected_split,
        dataset_manifest_hash=dataset_manifest_hash,
        git_commit=git_commit,
    )
    override_reason: str | None = None

    if resume and predictions_path.exists():
        existing_metadata = _load_run_metadata(metadata_path)
        if not run_metadata["strict_resume_supported"]:
            missing = ", ".join(str(item) for item in run_metadata["missing_components"])
            if not unsafe_resume_override:
                raise RuntimeError(
                    f"[{predictor.name}] strict resume is unsupported without fingerprint inputs: missing {missing}"
                )
            override_reason = f"missing_fingerprint_inputs:{missing}"
            _emit(f"[{predictor.name}] unsafe resume override enabled despite missing fingerprint inputs: {missing}")

        mismatched_fields = _fingerprint_mismatch_fields(run_metadata, existing_metadata)
        if mismatched_fields:
            details = ", ".join(mismatched_fields)
            if not unsafe_resume_override:
                raise RuntimeError(
                    f"[{predictor.name}] resume fingerprint mismatch for {details}; "
                    "rerun without --resume or use the explicit unsafe override"
                )
            override_reason = f"fingerprint_mismatch:{details}"
            _emit(f"[{predictor.name}] unsafe resume override ignoring fingerprint mismatch: {details}")

        existing = read_jsonl(predictions_path)
        schema_incompatible = False
        for row in existing:
            ex_id = _row_example_id(row)
            if not ex_id or ex_id not in target_ids or ex_id in completed_ids:
                continue
            if not _row_is_compatible(row):
                schema_incompatible = True
                break
            records.append(row)
            completed_ids.add(ex_id)

        if schema_incompatible:
            if not unsafe_resume_override:
                raise RuntimeError(
                    f"[{predictor.name}] resume data is incompatible with the flattened evaluator schema; "
                    "rerun without --resume or use the explicit unsafe override"
                )
            override_reason = "schema_incompatible_restart"
            _emit(f"[{predictor.name}] unsafe resume override restarting fresh outputs after schema mismatch")
            records = []
            completed_ids = set()
        elif completed_ids:
            _emit(f"[{predictor.name}] resume loaded {len(completed_ids)}/{total} predictions from {predictions_path}")

    if unsafe_resume_override and override_reason is not None:
        run_metadata = _build_run_metadata(
            predictor_name=str(getattr(predictor, "name", type(predictor).__name__)),
            resolved_config_hash=resolved_config_hash,
            selected_split=selected_split,
            dataset_manifest_hash=dataset_manifest_hash,
            git_commit=git_commit,
            unsafe_resume_override=True,
            override_reason=override_reason,
        )

    _write_run_metadata(metadata_path, run_metadata)

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
                protocol_category=_effective_protocol_category(ex),
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
                "protocol_category": _effective_protocol_category(ex),
                "vision_supported_target": _c2_vision_supported_answer(ex),
                "text_supported_target": _c2_text_supported_answer(ex),
                "vision_info_state": ex.vision_info_state,
                "text_info_state": ex.text_info_state,
                "pairwise_relation": ex.pairwise_relation,
                "joint_answer": ex.joint_answer,
                "final_answer": prediction.final_answer,
                "abstained": bool(prediction.abstained),
                "confidence": float(prediction.confidence or 0.0),
                "raw_output": None,
                "correct": bool(correct),
                "task_success": bool(task_success),
                "pred_action": None,
                "pred_conflict_type": None,
                "r_v": None,
                "r_t": None,
                "audit": None,
                "projection_succeeded": False,
                "used_fallback_dist": False,
                "parsed_unknown": False,
                "parsed_in_active_vocab": False,
                "canonicalized_candidate": None,
                "out_of_vocab_generation": False,
                "dist_argmax_label": None,
                "parsed_argmax_agree": False,
                "multimodal_raw_output": None,
                "vision_raw_output": None,
                "text_raw_output": None,
                "c2_vision_only_correct": None,
                "c2_text_only_correct": None,
                "c2_multimodal_abstained": None,
            }
            raw_output = prediction.raw_text
            if raw_output is None:
                metadata_raw = metadata.get("raw_output")
                if isinstance(metadata_raw, str):
                    raw_output = metadata_raw
            if raw_output is not None:
                row["raw_output"] = str(raw_output)
            for key in ("pred_action", "pred_conflict_type", "r_v", "r_t", "audit", *RAW_METADATA_KEYS):
                if key in metadata:
                    row[key] = metadata[key]
            for key in (*DIAGNOSTIC_BOOL_KEYS, *DIAGNOSTIC_VALUE_KEYS):
                if key in metadata:
                    row[key] = metadata[key]
            for key, value in _compute_c2_diagnostics(predictor, ex, prediction, canon_cfg).items():
                row[key] = value

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
