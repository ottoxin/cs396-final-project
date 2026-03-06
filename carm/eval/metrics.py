from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


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
    if pred_action:
        if oracle == "abstain":
            return abstained
        if oracle == "require_agreement":
            return pred_action == "require_agreement" and (abstained or correct)
        if oracle in {"trust_vision", "trust_text"}:
            return pred_action == oracle and correct
        return False

    category = str(protocol_category or "").strip().upper()

    if category == "C2":
        return abstained
    if category == "C5" or oracle == "abstain":
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
        "accuracy": accuracy(records),
        "coverage": coverage(records),
        "accuracy_on_answered": accuracy_on_answered(records),
        "task_success_per_category": _per_category_task_success(records),
        "task_success_per_split": _per_split_task_success(records),
        "accuracy_per_category": _per_category_accuracy(records),
        "risk_coverage_task_success": risk_coverage_curve_task_success(records),
        "example_counts_by_split": _counts_by(records, lambda row: _example_value(row, "split", "")),
        "example_counts_by_category": _counts_by(records, _protocol_category_value),
    }
