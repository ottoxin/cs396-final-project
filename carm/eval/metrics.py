from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


ACTION_LABELS = (
    "trust_vision",
    "trust_text",
    "require_agreement",
    "abstain",
)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _example_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row.get(key, default)
    example = row.get("example")
    if isinstance(example, dict):
        return example.get(key, default)
    return default


def _target_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row.get(key, default)
    targets = row.get("targets")
    if isinstance(targets, dict):
        return targets.get(key, default)
    return default


def _derived_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row.get(key, default)
    derived = row.get("derived")
    if isinstance(derived, dict):
        return derived.get(key, default)
    return default


def _prediction_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row.get(key, default)

    policy = row.get("policy_output")
    if isinstance(policy, dict) and key in policy:
        return policy.get(key, default)

    answer = row.get("answer_output")
    if isinstance(answer, dict):
        if key == "final_answer":
            return answer.get("raw_text", default)
        if key == "confidence":
            if "policy_confidence" in answer:
                return answer.get("policy_confidence", default)
            return answer.get("answer_confidence", default)

    return default


def _flag_value(row: dict[str, Any], key: str) -> bool:
    return _as_bool(_prediction_value(row, key, row.get(key, False)))


def _correct_value(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return _as_bool(row.get("correct"))

    for key in ("canonical_correct", "exact_correct", "semantic_correct"):
        value = _derived_value(row, key, None)
        if value is not None:
            return _as_bool(value)
    return False


def _confidence_value(row: dict[str, Any]) -> float:
    value = _prediction_value(row, "confidence", None)
    if value is None:
        value = _prediction_value(row, "policy_confidence", None)
    if value is None:
        value = _prediction_value(row, "answer_confidence", 0.0)
    return float(value or 0.0)


def _abstained_value(row: dict[str, Any]) -> bool:
    return _as_bool(_prediction_value(row, "abstained", False))


def _protocol_category_value(row: dict[str, Any]) -> str:
    category = str(_example_value(row, "protocol_category", "")).strip()
    if category:
        return category

    metadata = _example_value(row, "metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("protocol_category", "")).strip()
    return ""


def _final_answer_value(row: dict[str, Any]) -> str:
    return str(_prediction_value(row, "final_answer", row.get("final_answer", ""))).strip().lower()


def _action_pairs(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    valid = set(ACTION_LABELS)
    for row in records:
        oracle = str(_target_value(row, "oracle_action", "")).strip().lower()
        pred = _prediction_value(row, "pred_action", None)
        pred_text = str(pred).strip().lower() if pred is not None else ""
        if oracle in valid and pred_text in valid:
            pairs.append((oracle, pred_text))
    return pairs


def task_success_from_components(
    oracle_action: str,
    *args: object,
    protocol_category: str | None = None,
) -> bool:
    if len(args) == 2:
        pred_action = None
        abstained = _as_bool(args[0])
        correct = _as_bool(args[1])
    elif len(args) == 3:
        pred_action = str(args[0]).strip().lower() if args[0] is not None else ""
        abstained = _as_bool(args[1])
        correct = _as_bool(args[2])
    else:
        raise TypeError("task_success_from_components expects (oracle_action, abstained, correct) or legacy 4-arg form.")

    oracle = str(oracle_action).strip().lower()
    category = str(protocol_category or "").strip().upper()

    if category in {"C2", "C5"}:
        return abstained
    if category in {"C1", "C3", "C4"}:
        return (not abstained) and correct

    if oracle == "abstain":
        return abstained
    if oracle == "require_agreement":
        return abstained or correct
    if oracle in {"trust_vision", "trust_text"}:
        return (not abstained) and correct
    return False


def task_success_single(row: dict[str, Any]) -> bool:
    return task_success_from_components(
        str(_target_value(row, "oracle_action", "")),
        _prediction_value(row, "pred_action", None),
        _abstained_value(row),
        _correct_value(row),
        protocol_category=_protocol_category_value(row),
    )


def task_success_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if task_success_single(r) else 0.0 for r in records]))


def accuracy(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if _correct_value(r) else 0.0 for r in records]))


def coverage(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(np.mean([0.0 if _abstained_value(r) else 1.0 for r in records]))


def accuracy_on_answered(records: list[dict[str, Any]]) -> float:
    answered = [r for r in records if not _abstained_value(r)]
    if not answered:
        return 0.0
    return float(np.mean([1.0 if _correct_value(r) else 0.0 for r in answered]))


def action_accuracy(records: list[dict[str, Any]]) -> float | None:
    pairs = _action_pairs(records)
    if not pairs:
        return None
    return float(np.mean([1.0 if oracle == pred else 0.0 for oracle, pred in pairs]))


def action_macro_f1(records: list[dict[str, Any]]) -> float | None:
    pairs = _action_pairs(records)
    if not pairs:
        return None

    scores: list[float] = []
    for label in ACTION_LABELS:
        tp = sum(1 for oracle, pred in pairs if oracle == label and pred == label)
        fp = sum(1 for oracle, pred in pairs if oracle != label and pred == label)
        fn = sum(1 for oracle, pred in pairs if oracle == label and pred != label)
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(scores))


def _task_success_with_threshold(row: dict[str, Any], threshold: float) -> bool:
    effective_abstained = _abstained_value(row) or (_confidence_value(row) < threshold)
    pred_action = _prediction_value(row, "pred_action", None)
    return task_success_from_components(
        str(_target_value(row, "oracle_action", "")),
        pred_action,
        effective_abstained,
        _correct_value(row),
        protocol_category=_protocol_category_value(row),
    )


def risk_coverage_curve_task_success(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not records:
        return []

    thresholds = sorted({_confidence_value(r) for r in records}, reverse=True)
    thresholds = [max(thresholds) + 1e-12] + thresholds

    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        answered = [r for r in records if not (_abstained_value(r) or (_confidence_value(r) < threshold))]
        coverage_at_threshold = len(answered) / len(records)
        success = np.mean([1.0 if _task_success_with_threshold(r, threshold) else 0.0 for r in records])
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(coverage_at_threshold),
                "risk": float(1.0 - success),
            }
        )

    deduped: list[dict[str, float]] = []
    for point in curve:
        if deduped and abs(point["coverage"] - deduped[-1]["coverage"]) < 1e-12 and abs(point["risk"] - deduped[-1]["risk"]) < 1e-12:
            deduped[-1]["threshold"] = min(deduped[-1]["threshold"], point["threshold"])
            continue
        deduped.append(point)
    return deduped


def _mean_task_success(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if task_success_single(r) else 0.0 for r in records]))


def _flag_rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if _flag_value(r, key) else 0.0 for r in records]))


def _final_unknown_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if _final_answer_value(r) == "unknown" else 0.0 for r in records]))


def _optional_bool_mean(records: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in records:
        value = row.get(key)
        if value is None:
            continue
        values.append(1.0 if _as_bool(value) else 0.0)
    if not values:
        return None
    return float(np.mean(values))


def _c2_diagnostic_metric(records: list[dict[str, Any]], key: str) -> float | None:
    c2_rows = [row for row in records if _protocol_category_value(row) == "C2"]
    return _optional_bool_mean(c2_rows, key)


def _ordered_group_items(groups: dict[str, list[dict[str, Any]]], preferred: list[str]) -> list[tuple[str, list[dict[str, Any]]]]:
    ordered: list[tuple[str, list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for key in preferred:
        if key in groups:
            ordered.append((key, groups[key]))
            seen.add(key)
    for key in sorted(groups):
        if key not in seen:
            ordered.append((key, groups[key]))
    return ordered


def _per_category_task_success(records: list[dict[str, Any]]) -> dict[str, float]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        category = _protocol_category_value(row)
        if category:
            by_category[category].append(row)
    return {
        key: _mean_task_success(rows)
        for key, rows in _ordered_group_items(by_category, ["C1", "C2", "C3", "C4", "C5"])
    }


def _per_category_accuracy(records: list[dict[str, Any]]) -> dict[str, float]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        category = _protocol_category_value(row)
        if category:
            by_category[category].append(row)
    return {
        key: accuracy(rows)
        for key, rows in _ordered_group_items(by_category, ["C1", "C2", "C3", "C4", "C5"])
    }


def _per_category_flag_rate(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        category = _protocol_category_value(row)
        if category:
            by_category[category].append(row)
    return {
        group: _flag_rate(rows, key)
        for group, rows in _ordered_group_items(by_category, ["C1", "C2", "C3", "C4", "C5"])
    }


def _per_category_final_unknown_rate(records: list[dict[str, Any]]) -> dict[str, float]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        category = _protocol_category_value(row)
        if category:
            by_category[category].append(row)
    return {
        group: _final_unknown_rate(rows)
        for group, rows in _ordered_group_items(by_category, ["C1", "C2", "C3", "C4", "C5"])
    }


def _per_split_task_success(records: list[dict[str, Any]]) -> dict[str, float]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        split = str(_example_value(row, "split", "")).strip()
        if split:
            by_split[split].append(row)
    return {
        key: _mean_task_success(rows)
        for key, rows in _ordered_group_items(
            by_split,
            ["train", "val", "test_id", "test_ood_family", "test_ood_severity", "test_ood_hard_swap"],
        )
    }


def _counts_by(records: list[dict[str, Any]], value_fn) -> dict[str, int]:
    groups: dict[str, int] = defaultdict(int)
    for row in records:
        value = str(value_fn(row)).strip()
        if value:
            groups[value] += 1
    return dict(groups)


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_success": task_success_rate(records),
        "action_accuracy": action_accuracy(records),
        "action_macro_f1": action_macro_f1(records),
        "accuracy": accuracy(records),
        "coverage": coverage(records),
        "accuracy_on_answered": accuracy_on_answered(records),
        "task_success_per_category": _per_category_task_success(records),
        "task_success_per_split": _per_split_task_success(records),
        "accuracy_per_category": _per_category_accuracy(records),
        "projection_success_rate": _flag_rate(records, "projection_succeeded"),
        "projection_success_rate_per_category": _per_category_flag_rate(records, "projection_succeeded"),
        "fallback_rate": _flag_rate(records, "used_fallback_dist"),
        "fallback_rate_per_category": _per_category_flag_rate(records, "used_fallback_dist"),
        "parsed_unknown_rate": _flag_rate(records, "parsed_unknown"),
        "parsed_unknown_rate_per_category": _per_category_flag_rate(records, "parsed_unknown"),
        "parsed_in_active_vocab_rate": _flag_rate(records, "parsed_in_active_vocab"),
        "parsed_in_active_vocab_rate_per_category": _per_category_flag_rate(records, "parsed_in_active_vocab"),
        "out_of_vocab_generation_rate": _flag_rate(records, "out_of_vocab_generation"),
        "out_of_vocab_generation_rate_per_category": _per_category_flag_rate(records, "out_of_vocab_generation"),
        "parsed_argmax_agreement_rate": _flag_rate(records, "parsed_argmax_agree"),
        "parsed_argmax_agreement_rate_per_category": _per_category_flag_rate(records, "parsed_argmax_agree"),
        "final_unknown_rate": _final_unknown_rate(records),
        "final_unknown_rate_per_category": _per_category_final_unknown_rate(records),
        "c2_vision_only_accuracy": _c2_diagnostic_metric(records, "c2_vision_only_correct"),
        "c2_text_only_accuracy": _c2_diagnostic_metric(records, "c2_text_only_correct"),
        "c2_multimodal_abstention_rate": _c2_diagnostic_metric(records, "c2_multimodal_abstained"),
        "risk_coverage_task_success": risk_coverage_curve_task_success(records),
        "example_counts_by_split": _counts_by(records, lambda row: _example_value(row, "split", "")),
        "example_counts_by_category": _counts_by(records, _protocol_category_value),
    }
