from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from carm.data.schema import Family


def accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if r.get("correct", False) else 0.0 for r in records]))


def per_class_f1(records: list[dict], key_true: str, key_pred: str, labels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for r in records if r.get(key_true) == label and r.get(key_pred) == label)
        fp = sum(1 for r in records if r.get(key_true) != label and r.get(key_pred) == label)
        fn = sum(1 for r in records if r.get(key_true) == label and r.get(key_pred) != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[label] = 0.0 if precision + recall == 0.0 else float(2 * precision * recall / (precision + recall))
    return out


def macro_f1(records: list[dict], key_true: str, key_pred: str, labels: list[str]) -> float:
    f1 = per_class_f1(records, key_true, key_pred, labels)
    return float(np.mean(list(f1.values()))) if f1 else 0.0


def action_accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if r.get("oracle_action") == r.get("pred_action") else 0.0 for r in records]))


def risk_coverage_curve(records: list[dict]) -> list[dict[str, float]]:
    if not records:
        return []

    scored = sorted(records, key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
    out: list[dict[str, float]] = []
    retained: list[dict[str, Any]] = []
    total = len(scored)

    for i, rec in enumerate(scored, start=1):
        retained.append(rec)
        cov = i / total
        acc = np.mean([1.0 if r.get("correct", False) else 0.0 for r in retained])
        out.append({"coverage": round(float(cov), 4), "risk": round(1.0 - float(acc), 4)})
    return out


def expected_calibration_error(confidences: list[float], outcomes: list[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences)
    y = np.asarray(outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / len(conf)) * abs(float(np.mean(conf[mask])) - float(np.mean(y[mask])))
    return float(ece)


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences)
    y = np.asarray(outcomes)
    return float(np.mean((conf - y) ** 2))


def reliability_calibration(records: list[dict]) -> dict[str, float]:
    rv_conf = [float(r["r_v"]) for r in records]
    rt_conf = [float(r["r_t"]) for r in records]
    rv_out = [1 if float(r.get("target_r_v", 0.0)) >= 0.5 else 0 for r in records]
    rt_out = [1 if float(r.get("target_r_t", 0.0)) >= 0.5 else 0 for r in records]

    ece_r_v = expected_calibration_error(rv_conf, rv_out)
    ece_r_t = expected_calibration_error(rt_conf, rt_out)
    brier_r_v = brier_score(rv_conf, rv_out)
    brier_r_t = brier_score(rt_conf, rt_out)
    return {
        "ece_r_v": ece_r_v,
        "ece_r_t": ece_r_t,
        "brier_r_v": brier_r_v,
        "brier_r_t": brier_r_t,
        "ece": float((ece_r_v + ece_r_t) / 2.0),
        "brier": float((brier_r_v + brier_r_t) / 2.0),
    }


def monotonicity_violation_rate(records: list[dict]) -> float:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        mod = str(r.get("corrupt_modality", "none"))
        if mod == "none":
            continue
        key = (mod, str(r.get("operator", "unknown")))
        severity = int(r.get("severity", 0))
        rel = float(r["r_v"] if mod == "vision" else r["r_t"])
        grouped[key].append((severity, rel))

    violations = 0
    checks = 0
    for _, seq in grouped.items():
        by_sev: dict[int, list[float]] = defaultdict(list)
        for severity, rel in seq:
            by_sev[severity].append(rel)
        ordered = sorted((s, float(np.mean(v))) for s, v in by_sev.items())
        for i in range(1, len(ordered)):
            checks += 1
            if ordered[i][1] > ordered[i - 1][1] + 1e-6:
                violations += 1
    return float(violations / checks) if checks else 0.0


def _per_family_accuracy(records: list[dict]) -> dict[str, float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[str(row.get("family", "none"))].append(row)
    return {k: accuracy(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}


def _per_split_accuracy(records: list[dict]) -> dict[str, float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[str(row.get("split", "train"))].append(row)
    return {k: accuracy(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}


def summarize_metrics(records: list[dict]) -> dict[str, Any]:
    labels = [Family.NONE.value, Family.EXISTENCE.value, Family.COUNT.value, Family.ATTRIBUTE_COLOR.value]

    out: dict[str, Any] = {
        "accuracy": accuracy(records),
        "action_accuracy": action_accuracy(records),
        "macro_f1_conflict": macro_f1(records, "family", "pred_conflict_type", labels),
        "per_type_f1": per_class_f1(records, "family", "pred_conflict_type", labels),
        "risk_coverage": risk_coverage_curve(records),
        "monotonicity_violation_rate": monotonicity_violation_rate(records),
        "accuracy_per_family": _per_family_accuracy(records),
        "accuracy_per_split": _per_split_accuracy(records),
    }
    out.update(reliability_calibration(records))

    consistent = [r for r in records if r.get("family") == Family.NONE.value]
    conflict = [r for r in records if r.get("family") != Family.NONE.value]
    out["accuracy_consistent"] = accuracy(consistent)
    out["accuracy_conflict"] = accuracy(conflict)

    split_counts = Counter(str(r.get("split", "unknown")) for r in records)
    out["example_counts_by_split"] = dict(sorted(split_counts.items(), key=lambda kv: kv[0]))
    return out
