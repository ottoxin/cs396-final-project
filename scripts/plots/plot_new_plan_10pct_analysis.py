#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization" / "scripts"
DEFAULT_EXPERIMENTAL_ROOT = PROJECT_ROOT / "outputs" / "experimental" / "RUN-EXP-0007_10pct_qwen_protocol"
DEFAULT_CONTROL_ROOT = PROJECT_ROOT / "outputs" / "carm" / "RUN-CTRL-0001_10pct_protocol"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "RUN-ANALYSIS-0001_10pct_protocol"

PROTOCOL_ROWS = [
    ("C1", "informative", "informative", "consistent", "require_agreement", "gold_answer"),
    ("C2", "informative", "uninformative", "asymmetric", "trust_vision", "gold_answer"),
    ("C3", "uninformative", "informative", "asymmetric", "trust_text", "gold_answer"),
    ("C4", "informative", "informative", "contradictory", "abstain", "<ABSTAIN>"),
    ("C5", "uninformative", "uninformative", "both_weak", "abstain", "<ABSTAIN>"),
]
PROTOCOL_COLORS = {
    "C1": "#D8F3DC",
    "C2": "#FEECC8",
    "C3": "#DDEAF7",
    "C4": "#F6DDF0",
    "C5": "#E9ECEF",
}
CATEGORY_ORDER = ["C1", "C2", "C3", "C4", "C5"]
RELATION_ORDER = ["consistent", "contradictory", "asymmetric", "both_weak"]
ACTION_ORDER = ["require_agreement", "abstain", "trust_vision", "trust_text"]
MODEL_COLORS = {
    "backbone_direct": "#355070",
    "agreement_check": "#0E8A6A",
    "confidence_threshold": "#D9A404",
    "probe_heuristic": "#7B61A8",
    "prompt_only_abstain": "#C65D2E",
    "old_action_only": "#1F3A5F",
    "structured_carm": "#B23A48",
}


def _load_visualization_helpers() -> tuple[callable, callable, callable]:
    if SKILL_SCRIPTS_DIR.exists():
        sys.path.insert(0, str(SKILL_SCRIPTS_DIR))
        from figure_export import save_publication_figure
        from style_presets import apply_publication_style, set_color_palette

        return apply_publication_style, set_color_palette, save_publication_figure
    raise RuntimeError(
        "Visualization skill helpers not found. Expected scripts under "
        f"{SKILL_SCRIPTS_DIR}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the non-ablation figure bundle for the completed 10 percent NEW_PLAN stage.")
    parser.add_argument("--experimental-root", type=Path, default=DEFAULT_EXPERIMENTAL_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    if key in metrics and metrics[key] is not None:
        return float(metrics[key])
    if key == "answer_accuracy" and metrics.get("accuracy") is not None:
        return float(metrics["accuracy"])
    if key == "task_success_revised" and metrics.get("task_success") is not None:
        return float(metrics["task_success"])
    raise KeyError(key)


def _optional_metric_value(metrics: dict[str, Any], key: str) -> float | None:
    try:
        return _metric_value(metrics, key)
    except KeyError:
        return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    _, _, save_publication_figure = _load_visualization_helpers()
    save_publication_figure(fig, output_base, formats=["png"], dpi=300)
    plt.close(fig)


def _apply_style() -> None:
    apply_publication_style, set_color_palette, _ = _load_visualization_helpers()
    apply_publication_style("default")
    set_color_palette("okabe_ito")


def _stacked_rate(values: Counter, total: int, key: str) -> float:
    return float(values.get(key, 0) / max(1, total))


def _legacy_to_hf_category(category: str) -> str | None:
    mapping = {
        "C1": "C1",
        "C2": "C4",
        "C3": "C2",
        "C4": "C3",
        "C5": "C5",
    }
    return mapping.get(category)


def _canonical_protocol_category(row: dict[str, Any]) -> str | None:
    relation = str(
        row.get("derived_pairwise_relation")
        or row.get("pairwise_relation")
        or ""
    ).strip().lower()
    vision_info = str(
        row.get("derived_vision_info_state")
        or row.get("vision_info_state")
        or ""
    ).strip().lower()
    text_info = str(
        row.get("derived_text_info_state")
        or row.get("text_info_state")
        or ""
    ).strip().lower()
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

    action = str(
        row.get("derived_action_target")
        or row.get("oracle_action")
        or row.get("gold_action_legacy")
        or ""
    ).strip().lower()
    vision_target = row.get("vision_supported_target")
    text_target = row.get("text_supported_target")
    if action == "require_agreement":
        return "C1"
    if action == "trust_vision":
        return "C2"
    if action == "trust_text":
        return "C3"
    if action == "abstain":
        if vision_target is not None and text_target is not None:
            return "C4"
        return "C5"

    raw_category = str(row.get("protocol_category") or "").strip().upper()
    return _legacy_to_hf_category(raw_category)


def _group_mean(rows: list[dict[str, Any]], metric_key: str) -> dict[str, float]:
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        category = _canonical_protocol_category(row)
        if category not in CATEGORY_ORDER:
            continue
        value = row.get(metric_key)
        if value is None:
            continue
        by_category[category].append(1.0 if _boolish(value) else float(value))
    return {
        category: float(np.mean(by_category.get(category, [0.0])))
        for category in CATEGORY_ORDER
    }


def _build_agreement_stats(backbone_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_category: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    for row in backbone_rows:
        category = _canonical_protocol_category(row)
        if category not in CATEGORY_ORDER:
            continue
        totals[category] += 1
        vision_answer = row.get("vision_answer")
        text_answer = row.get("text_answer")
        vision_ok = _boolish(row.get("vision_matches_vision_target"))
        text_ok = _boolish(row.get("text_matches_text_target"))
        if vision_answer == text_answer:
            by_category[category]["agree"] += 1
        else:
            by_category[category]["disagree"] += 1
        if vision_ok and text_ok:
            by_category[category]["both_correct"] += 1
        elif vision_ok and not text_ok:
            by_category[category]["only_vision_correct"] += 1
        elif text_ok and not vision_ok:
            by_category[category]["only_text_correct"] += 1
        else:
            by_category[category]["neither_correct"] += 1

    stats: dict[str, dict[str, float]] = {}
    for category in CATEGORY_ORDER:
        total = totals[category]
        counts = by_category[category]
        stats[category] = {
            "agree_rate": _stacked_rate(counts, total, "agree"),
            "disagree_rate": _stacked_rate(counts, total, "disagree"),
            "both_correct_rate": _stacked_rate(counts, total, "both_correct"),
            "only_vision_correct_rate": _stacked_rate(counts, total, "only_vision_correct"),
            "only_text_correct_rate": _stacked_rate(counts, total, "only_text_correct"),
            "neither_correct_rate": _stacked_rate(counts, total, "neither_correct"),
            "n": int(total),
        }
    return stats


def _plot_label_schema(output_dir: Path) -> dict[str, Any]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 3.5))
    ax.axis("off")

    col_labels = ["Category", "Vision info", "Text info", "Relation", "Joint action", "Joint answer"]
    cell_text = [[*row] for row in PROTOCOL_ROWS]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#243B53")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            continue
        category = PROTOCOL_ROWS[row_idx - 1][0]
        cell.set_facecolor(PROTOCOL_COLORS.get(category, "white"))
        if col_idx == 0:
            cell.get_text().set_weight("bold")

    ax.set_title("Figure 1. Revised Five-Category Label Schema", fontsize=10, pad=8)
    output_base = output_dir / "figures" / "fig01_label_schema_summary"
    _save_figure(fig, output_base)
    return {"title": "Figure 1. Revised Five-Category Label Schema", "path": str(output_base.with_suffix(".png"))}


def _plot_unimodal_agreement(backbone_rows: list[dict[str, Any]], output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    stats = _build_agreement_stats(backbone_rows)
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), width_ratios=[1.6, 1.0])
    x = np.arange(len(CATEGORY_ORDER))

    buckets = [
        ("both_correct_rate", "Both correct", "#2A9D8F"),
        ("only_vision_correct_rate", "Only vision correct", "#457B9D"),
        ("only_text_correct_rate", "Only text correct", "#E9C46A"),
        ("neither_correct_rate", "Neither correct", "#8D99AE"),
    ]
    bottoms = np.zeros(len(CATEGORY_ORDER))
    for key, label, color in buckets:
        values = np.array([stats[c][key] for c in CATEGORY_ORDER])
        axes[0].bar(x, values, bottom=bottoms, color=color, edgecolor="white", linewidth=0.6, label=label)
        bottoms += values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CATEGORY_ORDER)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Share of examples")
    axes[0].set_title("Correctness buckets by category")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    agree = [stats[c]["agree_rate"] for c in CATEGORY_ORDER]
    disagree = [stats[c]["disagree_rate"] for c in CATEGORY_ORDER]
    axes[1].plot(x, agree, marker="o", color="#264653", linewidth=1.8, label="Vision/text agree")
    axes[1].plot(x, disagree, marker="s", color="#D62828", linewidth=1.5, label="Vision/text disagree")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CATEGORY_ORDER)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Agreement rate by category")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=False)

    fig.suptitle("Figure 2. Unimodal Agreement Behavior by Category", fontsize=10, y=1.03)
    output_base = output_dir / "figures" / "fig02_unimodal_agreement_behavior"
    _save_figure(fig, output_base)
    return (
        {"title": "Figure 2. Unimodal Agreement Behavior by Category", "path": str(output_base.with_suffix(".png"))},
        stats,
    )


def _plot_entropy_distributions(backbone_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), sharey=False)
    metric_specs = [
        ("multimodal_entropy", "Multimodal entropy", "#355070"),
        ("vision_entropy", "Vision entropy", "#457B9D"),
        ("text_entropy", "Text entropy", "#E76F51"),
    ]
    for ax, (metric_key, title, color) in zip(axes, metric_specs, strict=True):
        values = []
        for category in CATEGORY_ORDER:
            category_values = [
                _safe_float(row.get(metric_key))
                for row in backbone_rows
                if _canonical_protocol_category(row) == category
            ]
            values.append([v for v in category_values if v is not None])
        bp = ax.boxplot(
            values,
            patch_artist=True,
            tick_labels=CATEGORY_ORDER,
            widths=0.65,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("white")
        for median in bp["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.0)
        ax.set_title(title)
        ax.set_xlabel("Category")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Entropy")
    fig.suptitle("Figure 3. Confidence / Entropy by Category", fontsize=10, y=1.02)
    output_base = output_dir / "figures" / "fig03_entropy_by_category"
    _save_figure(fig, output_base)
    return {"title": "Figure 3. Confidence / Entropy by Category", "path": str(output_base.with_suffix(".png"))}


def _plot_baseline_comparison(
    backbone_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, dict[str, Any]],
    structured_metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), height_ratios=[1.0, 1.4])

    per_category_accuracy = _group_mean(backbone_rows, "correct")
    overall_answer_accuracy = float(np.mean([1.0 if _boolish(row.get("correct")) else 0.0 for row in backbone_rows]))
    categories = CATEGORY_ORDER + ["overall"]
    values = [float(per_category_accuracy[c]) for c in CATEGORY_ORDER] + [overall_answer_accuracy]
    colors = [PROTOCOL_COLORS[c] for c in CATEGORY_ORDER] + ["#355070"]
    axes[0].bar(np.arange(len(categories)), values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_xticks(np.arange(len(categories)))
    axes[0].set_xticklabels(categories)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Answer accuracy")
    axes[0].set_title("Panel A. Backbone-direct answer accuracy")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    model_order = ["prompt_only_abstain", "agreement_check", "confidence_threshold", "structured_carm"]
    model_labels = {
        "prompt_only_abstain": "prompt_only_abstain",
        "agreement_check": "agreement_check",
        "confidence_threshold": "confidence_threshold",
        "structured_carm": "structured_carm",
    }
    metric_order = ["coverage", "task_success_revised", "accuracy_on_answered"]
    metric_titles = {
        "coverage": "Coverage",
        "task_success_revised": "Task success",
        "accuracy_on_answered": "Accuracy on answered",
    }
    x = np.arange(len(metric_order))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(model_order))
    source_lookup = {
        "prompt_only_abstain": baseline_metrics["prompt_only_abstain"],
        "agreement_check": baseline_metrics["agreement_check"],
        "confidence_threshold": baseline_metrics["confidence_threshold"],
        "structured_carm": structured_metrics,
    }
    for offset, model_key in zip(offsets, model_order, strict=True):
        vals = [_metric_value(source_lookup[model_key], metric) for metric in metric_order]
        axes[1].bar(
            x + offset,
            vals,
            width=width,
            label=model_labels[model_key],
            color=MODEL_COLORS[model_key],
            edgecolor="white",
            linewidth=0.8,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([metric_titles[m] for m in metric_order])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Panel B. Selective baselines and structured CARM")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    fig.suptitle("Figure 4. Baseline Comparison", fontsize=10, y=1.01)
    output_base = output_dir / "figures" / "fig04_baseline_comparison"
    _save_figure(fig, output_base)
    return {"title": "Figure 4. Baseline Comparison", "path": str(output_base.with_suffix(".png"))}


def _plot_backbone_direct_category_panels(backbone_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.2), sharex=True)

    categories = CATEGORY_ORDER
    x = np.arange(len(categories))
    answer_accuracy = [float(_group_mean(backbone_rows, "correct")[c]) for c in categories]
    task_success = [float(_group_mean(backbone_rows, "task_success_revised")[c]) for c in categories]
    overall_answer_accuracy = float(np.mean([1.0 if _boolish(row.get("correct")) else 0.0 for row in backbone_rows]))
    overall_task_success = float(np.mean([1.0 if _boolish(row.get("task_success_revised")) else 0.0 for row in backbone_rows]))

    base_bar_color = "#4EA1D3"
    task_bar_colors = ["#E9A000" if c in {"C1", "C2", "C3"} else "#F4C15D" for c in categories]
    task_hatches = ["", "", "", "//", "//"]

    panels = [
        (
            axes[0],
            answer_accuracy,
            "Backbone Direct Baseline: Per-Category Answer Accuracy on test_id",
            "Accuracy",
            overall_answer_accuracy,
            [base_bar_color] * len(categories),
            [""] * len(categories),
            None,
        ),
        (
            axes[1],
            task_success,
            "Backbone Direct Baseline: Per-Category Task Success on test_id",
            "Task Success",
            overall_task_success,
            task_bar_colors,
            task_hatches,
            "C4/C5 require abstention; this panel is diagnostic only for an answer-forcing baseline.",
        ),
    ]

    for ax, values, title, ylabel, overall, colors, hatches, footnote in panels:
        bars = ax.bar(x, values, width=0.66, color=colors, edgecolor="#2F4F4F", linewidth=0.9)
        for bar, hatch in zip(bars, hatches, strict=True):
            if hatch:
                bar.set_hatch(hatch)
        ax.axhline(overall, color="#8A8A8A", linestyle="--", linewidth=1.0)
        ax.text(len(categories) - 0.1, overall + 0.015, f"overall = {overall:.3f}", ha="right", va="bottom", fontsize=8, color="#666666")
        for idx, value in enumerate(values):
            ax.text(idx, min(0.985, value + 0.018), f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        if footnote:
            ax.text(0.99, -0.24, footnote, transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#666666")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_xlabel("Protocol Category")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_xlabel("Protocol Category")

    fig.suptitle("Diagnostic. Backbone-Direct Category Panels", fontsize=10, y=0.995)
    fig.tight_layout()
    output_base = output_dir / "figures" / "diag_backbone_direct_category_panels"
    _save_figure(fig, output_base)
    caption = "\n".join(
        [
            "# Diagnostic. Backbone-Direct Category Panels",
            "",
            "- Top panel: per-category answer accuracy for `backbone_direct` on `test_id`.",
            "- Bottom panel: per-category task success under HF semantics.",
            "- `C4` and `C5` are abstention-required, so the task-success panel is diagnostic rather than a fair main-comparison figure for an answer-forcing baseline.",
            "",
        ]
    )
    (output_dir / "figures" / "diag_backbone_direct_category_panels.md").write_text(caption, encoding="utf-8")
    return {
        "title": "Diagnostic. Backbone-Direct Category Panels",
        "path": str(output_base.with_suffix(".png")),
    }


def _plot_architecture(output_dir: Path) -> dict[str, Any]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.58, 0.12, 0.17, "Image"),
        (0.03, 0.33, 0.12, 0.17, "Caption"),
        (0.03, 0.08, 0.12, 0.17, "Question"),
        (0.22, 0.24, 0.18, 0.36, "Frozen Qwen\nmultimodal backbone"),
        (0.48, 0.62, 0.16, 0.12, "Vision-only\ncandidate + entropy"),
        (0.48, 0.44, 0.16, 0.12, "Text-only\ncandidate + entropy"),
        (0.48, 0.26, 0.16, 0.12, "Multimodal\ncandidate + entropy"),
        (0.72, 0.30, 0.12, 0.30, "Shared\nMLP trunk"),
        (0.88, 0.69, 0.1, 0.09, "Vision info"),
        (0.88, 0.56, 0.1, 0.09, "Text info"),
        (0.88, 0.43, 0.1, 0.09, "Relation"),
        (0.88, 0.30, 0.1, 0.09, "Action"),
    ]
    for x, y, w, h, label in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#243B53",
            facecolor="#F8F9FA",
        )
        ax.add_patch(rect)
        ax.text(x + (w / 2), y + (h / 2), label, ha="center", va="center", fontsize=8)

    arrows = [
        ((0.15, 0.66), (0.22, 0.48)),
        ((0.15, 0.41), (0.22, 0.42)),
        ((0.15, 0.16), (0.22, 0.36)),
        ((0.40, 0.48), (0.48, 0.68)),
        ((0.40, 0.48), (0.48, 0.50)),
        ((0.40, 0.48), (0.48, 0.32)),
        ((0.64, 0.68), (0.72, 0.52)),
        ((0.64, 0.50), (0.72, 0.46)),
        ((0.64, 0.32), (0.72, 0.40)),
        ((0.84, 0.60), (0.88, 0.73)),
        ((0.84, 0.54), (0.88, 0.60)),
        ((0.84, 0.48), (0.88, 0.47)),
        ((0.84, 0.42), (0.88, 0.34)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#243B53", lw=1.0))

    ax.text(0.62, 0.08, "Router reads modality evidence, relation, and action head", fontsize=8, ha="center")
    ax.set_title("Figure 5. Structured CARM Architecture", fontsize=10, pad=8)
    output_base = output_dir / "figures" / "fig05_structured_carm_architecture"
    _save_figure(fig, output_base)
    return {"title": "Figure 5. Structured CARM Architecture", "path": str(output_base.with_suffix(".png"))}


def _plot_learned_model_comparison(
    control_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    x = np.arange(len(CATEGORY_ORDER))
    width = 0.34
    control_vals = [float(_group_mean(control_rows, "task_success")[c]) for c in CATEGORY_ORDER]
    structured_vals = [float(_group_mean(structured_rows, "task_success_revised")[c]) for c in CATEGORY_ORDER]
    ax.bar(x - (width / 2), control_vals, width=width, color=MODEL_COLORS["old_action_only"], label="old_action_only", edgecolor="white")
    ax.bar(x + (width / 2), structured_vals, width=width, color=MODEL_COLORS["structured_carm"], label="structured_carm", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Task success")
    ax.set_xlabel("Protocol category")
    ax.set_title("Figure 6. Per-Category Learned-Model Results")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    output_base = output_dir / "figures" / "fig06_learned_model_per_category"
    _save_figure(fig, output_base)
    return {"title": "Figure 6. Per-Category Learned-Model Results", "path": str(output_base.with_suffix(".png"))}


def _confusion_matrix(rows: list[dict[str, Any]], truth_key: str, pred_key: str, labels: list[str]) -> np.ndarray:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for row in rows:
        truth = row.get(truth_key)
        pred = row.get(pred_key)
        if truth not in index or pred not in index:
            continue
        matrix[index[truth], index[pred]] += 1.0
    return matrix


def _plot_confusions(structured_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    _apply_style()
    relation_matrix = _confusion_matrix(structured_rows, "derived_pairwise_relation", "pred_pairwise_relation", RELATION_ORDER)
    action_matrix = _confusion_matrix(structured_rows, "derived_action_target", "pred_action", ACTION_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, matrix, labels, title in [
        (axes[0], relation_matrix, RELATION_ORDER, "Relation confusion"),
        (axes[1], action_matrix, ACTION_ORDER, "Action confusion"),
    ]:
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(title)
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#111111")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 7. Confusion Matrices for Structured CARM", fontsize=10, y=1.03)
    output_base = output_dir / "figures" / "fig07_confusion_matrices"
    _save_figure(fig, output_base)
    return {"title": "Figure 7. Confusion Matrices for Structured CARM", "path": str(output_base.with_suffix(".png"))}


def _select_failure_examples(structured_rows: list[dict[str, Any]], failure_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_id = {row["example_id"]: row for row in structured_rows}

    def choose(category: str, predicate) -> dict[str, Any] | None:
        candidates = [row for row in failure_rows if _canonical_protocol_category(row) == category and predicate(row)]
        candidates.sort(key=lambda row: float(row.get("multimodal_confidence", "0") or 0.0), reverse=True)
        for row in candidates:
            full = by_id.get(row["example_id"])
            if full is not None:
                return full
        return None

    chosen = [
        choose("C4", lambda row: row.get("failure_type") in {"c2_contradiction_collapse", "c4_contradiction_collapse"}),
        choose("C2", lambda row: "revised_task_failure" in str(row.get("failure_reason", ""))),
        choose("C5", lambda row: str(row.get("pred_action")) in {"trust_vision", "trust_text"}),
    ]
    return [row for row in chosen if row is not None]


def _plot_failure_examples(structured_rows: list[dict[str, Any]], failure_rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    examples = _select_failure_examples(structured_rows, failure_rows)
    _apply_style()
    fig, axes = plt.subplots(len(examples), 1, figsize=(10.4, 3.2 * max(1, len(examples))))
    if len(examples) == 1:
        axes = [axes]
    summary_rows = []
    for ax, row in zip(axes, examples, strict=True):
        ax.axis("off")
        summary = textwrap.dedent(
            f"""
            {_canonical_protocol_category(row)} | {row['example_id']}
            Q: {row['question']}
            vision={row.get('vision_answer')} | text={row.get('text_answer')} | multimodal={row.get('multimodal_answer')}
            gold relation={row.get('derived_pairwise_relation')} -> predicted relation={row.get('pred_pairwise_relation')}
            gold action={row.get('derived_action_target')} -> predicted action={row.get('pred_action')}
            gold info=(vision {row.get('derived_vision_info_state')}, text {row.get('derived_text_info_state')})
            pred info=(vision {row.get('pred_vision_info_state')}, text {row.get('pred_text_info_state')})
            confidence={float(row.get('multimodal_confidence') or 0.0):.3f} | revised_task_success={row.get('task_success_revised')}
            """
        ).strip()
        ax.text(
            0.01,
            0.98,
            summary,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#CED4DA"),
        )
        summary_rows.append(
            {
                "protocol_category": _canonical_protocol_category(row),
                "example_id": row["example_id"],
                "question": row["question"],
                "vision_answer": row.get("vision_answer"),
                "text_answer": row.get("text_answer"),
                "multimodal_answer": row.get("multimodal_answer"),
                "pred_action": row.get("pred_action"),
                "derived_action_target": row.get("derived_action_target"),
                "pred_pairwise_relation": row.get("pred_pairwise_relation"),
                "derived_pairwise_relation": row.get("derived_pairwise_relation"),
            }
        )
    fig.suptitle("Figure 8. Failure Diagnostic Examples", fontsize=10, y=0.995)
    output_base = output_dir / "figures" / "fig08_failure_examples"
    _save_figure(fig, output_base)
    markdown_lines = [
        "# Figure 8 Failure Examples",
        "",
    ]
    for row in summary_rows:
        markdown_lines.extend(
            [
                f"## {row['protocol_category']} {row['example_id']}",
                f"- question: {row['question']}",
                f"- answers: vision={row['vision_answer']}, text={row['text_answer']}, multimodal={row['multimodal_answer']}",
                f"- relation: gold={row['derived_pairwise_relation']}, predicted={row['pred_pairwise_relation']}",
                f"- action: gold={row['derived_action_target']}, predicted={row['pred_action']}",
                "",
            ]
        )
    (output_dir / "figures" / "fig08_failure_examples.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return {"title": "Figure 8. Failure Diagnostic Examples", "path": str(output_base.with_suffix(".png"))}


def _plot_c4_candidate_behavior(backbone_rows: list[dict[str, Any]], structured_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> list[float]:
        subset = [row for row in rows if _canonical_protocol_category(row) == "C4"]
        counts = Counter()
        for row in subset:
            if _boolish(row.get("abstained")):
                counts["abstain"] += 1
            elif row.get("multimodal_answer") == row.get("vision_supported_target"):
                counts["match_vision"] += 1
            elif row.get("multimodal_answer") == row.get("text_supported_target"):
                counts["match_text"] += 1
            else:
                counts["other"] += 1
        total = max(1, len(subset))
        return [counts[key] / total for key in ("abstain", "match_vision", "match_text", "other")]

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    categories = ["abstain", "match vision", "match text", "other"]
    for ax, title, values, color in [
        (axes[0], "backbone_direct", summarize(backbone_rows), MODEL_COLORS["backbone_direct"]),
        (axes[1], "structured_carm", summarize(structured_rows), MODEL_COLORS["structured_carm"]),
    ]:
        ax.bar(np.arange(len(categories)), values, color=color, edgecolor="white")
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Share of C4 examples")
    fig.suptitle("Diagnostic. Candidate-Answer Behavior for C4", fontsize=10, y=1.02)
    output_base = output_dir / "figures" / "diag_c4_candidate_behavior"
    _save_figure(fig, output_base)
    return {"title": "Diagnostic. Candidate-Answer Behavior for C4", "path": str(output_base.with_suffix(".png"))}


def _build_summary_markdown(
    output_dir: Path,
    *,
    agreement_stats: dict[str, dict[str, float]],
    backbone_metrics: dict[str, Any],
    agreement_metrics: dict[str, Any],
    confidence_metrics: dict[str, Any],
    prompt_metrics: dict[str, Any],
    probe_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    structured_metrics: dict[str, Any],
) -> None:
    lines = [
        "# 10 Percent Stage Summary",
        "",
        "## Main Result",
        "",
        f"- Old action-only control beats the structured four-head model on final decision quality in this 10 percent run: control task_success={control_metrics['task_success']:.4f}, control action_accuracy={control_metrics['action_accuracy']:.4f}; structured task_success_revised={structured_metrics['task_success_revised']:.4f}, structured action_accuracy={structured_metrics['action_accuracy']:.4f}.",
        f"- The structured model does learn some intermediate structure, especially vision informativeness (vision_info_accuracy={structured_metrics['vision_info_accuracy']:.4f}), but text_info_accuracy={structured_metrics['text_info_accuracy']:.4f} and relation_accuracy={structured_metrics['relation_accuracy']:.4f} remain weak.",
        "",
        "## Baseline Snapshot",
        "",
        f"- backbone_direct: answer_accuracy={_metric_value(backbone_metrics, 'answer_accuracy'):.4f}, task_success_revised={_metric_value(backbone_metrics, 'task_success_revised'):.4f}.",
        f"- agreement_check: coverage={agreement_metrics['coverage']:.4f}, task_success_revised={_metric_value(agreement_metrics, 'task_success_revised'):.4f}.",
        f"- confidence_threshold: coverage={confidence_metrics['coverage']:.4f}, task_success_revised={_metric_value(confidence_metrics, 'task_success_revised'):.4f}.",
        f"- prompt_only_abstain: coverage={prompt_metrics['coverage']:.4f}, task_success_revised={_metric_value(prompt_metrics, 'task_success_revised'):.4f}, action_accuracy={_format_metric(_optional_metric_value(prompt_metrics, 'action_accuracy'))}.",
        f"- probe_heuristic: coverage={probe_metrics['coverage']:.4f}, task_success_revised={_metric_value(probe_metrics, 'task_success_revised'):.4f}.",
        "",
        "## Diagnostic Answers",
        "",
        "- Q1. Do the new labels behave sensibly? Yes. The prepared data audit, explicit C1-C5 mapping, and the completed runs are consistent with the revised semantics.",
        "- Q2. Do simple baselines separate contradiction from irrelevance? Partly. Agreement-based abstention helps on contradictory rows, but direct multimodal answering still collapses to one side too often in C4.",
        "- Q3. Does the structured model improve over the old action-only head? No on this 10 percent run. The action-only control is stronger on final action/task-success metrics.",
        "- Q4. Which intermediate target helps most? Deferred. Ablations were explicitly held out for this stage.",
        f"- Q5. Does the structured model especially improve C4 and C5? No. Contradiction-category multimodal abstain rate is only {structured_metrics.get('contradiction_multimodal_abstain_rate', structured_metrics.get('c2_multimodal_abstain_rate', 0.0)):.4f}, and C5 remains difficult despite high abstention.",
        "- Q6. Are current features sufficient? Not yet. The structured head learns vision informativeness more readily than text informativeness or pairwise relation, which points to feature or objective limitations rather than label incoherence.",
        "",
        "## Unimodal Agreement Snapshot",
        "",
    ]
    for category in CATEGORY_ORDER:
        row = agreement_stats[category]
        lines.append(
            f"- {category}: agree={row['agree_rate']:.3f}, both_correct={row['both_correct_rate']:.3f}, only_vision_correct={row['only_vision_correct_rate']:.3f}, only_text_correct={row['only_text_correct_rate']:.3f}, neither={row['neither_correct_rate']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Remaining Work",
            "",
            "- Ablations remain intentionally deferred.",
            "- The next highest-value step is to inspect the generated figures and then decide whether to improve the structured feature set or supervision weighting before running ablations.",
            "",
        ]
    )
    (output_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    experimental_root = args.experimental_root.resolve()
    control_root = args.control_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    backbone_metrics = _load_json(experimental_root / "baselines" / "backbone_direct" / "metrics.json")
    agreement_metrics = _load_json(experimental_root / "baselines" / "agreement_check" / "metrics.json")
    confidence_metrics = _load_json(experimental_root / "baselines" / "confidence_threshold" / "metrics.json")
    prompt_metrics = _load_json(experimental_root / "baselines" / "prompt_only_abstain" / "metrics.json")
    probe_metrics = _load_json(experimental_root / "baselines" / "probe_heuristic" / "metrics.json")
    structured_metrics = _load_json(experimental_root / "structured_carm" / "test" / "metrics.json")
    control_metrics = _load_json(control_root / "eval_test_id" / "metrics.json")

    backbone_rows = _load_jsonl(experimental_root / "baselines" / "backbone_direct" / "per_example_predictions.jsonl")
    structured_rows = _load_jsonl(experimental_root / "structured_carm" / "test" / "per_example_predictions.jsonl")
    control_rows = _load_jsonl(control_root / "eval_test_id" / "per_example_predictions.jsonl")
    failure_rows = _load_csv(experimental_root / "structured_carm" / "test" / "failure_diagnostics.csv")

    figures = []
    fig, agreement_stats = _plot_unimodal_agreement(backbone_rows, output_dir)
    figures.append(_plot_label_schema(output_dir))
    figures.append(fig)
    figures.append(_plot_entropy_distributions(backbone_rows, output_dir))
    figures.append(_plot_baseline_comparison(backbone_rows, {
        "agreement_check": agreement_metrics,
        "confidence_threshold": confidence_metrics,
        "prompt_only_abstain": prompt_metrics,
    }, structured_metrics, output_dir))
    figures.append(_plot_backbone_direct_category_panels(backbone_rows, output_dir))
    figures.append(_plot_architecture(output_dir))
    figures.append(_plot_learned_model_comparison(control_rows, structured_rows, output_dir))
    figures.append(_plot_confusions(structured_rows, output_dir))
    figures.append(_plot_failure_examples(structured_rows, failure_rows, output_dir))
    figures.append(_plot_c4_candidate_behavior(backbone_rows, structured_rows, output_dir))

    manifest = {
        "experimental_root": str(experimental_root),
        "control_root": str(control_root),
        "figures": figures,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics_snapshot = {
        "backbone_direct": {
            "answer_accuracy": _metric_value(backbone_metrics, "answer_accuracy"),
            "task_success_revised": _metric_value(backbone_metrics, "task_success_revised"),
        },
        "agreement_check": {
            "coverage": agreement_metrics["coverage"],
            "task_success_revised": _metric_value(agreement_metrics, "task_success_revised"),
        },
        "confidence_threshold": {
            "coverage": confidence_metrics["coverage"],
            "task_success_revised": _metric_value(confidence_metrics, "task_success_revised"),
        },
        "prompt_only_abstain": {
            "coverage": prompt_metrics["coverage"],
            "task_success_revised": _metric_value(prompt_metrics, "task_success_revised"),
            "action_accuracy": _optional_metric_value(prompt_metrics, "action_accuracy"),
        },
        "probe_heuristic": {
            "coverage": probe_metrics["coverage"],
            "task_success_revised": _metric_value(probe_metrics, "task_success_revised"),
        },
        "old_action_only": {
            "task_success": control_metrics["task_success"],
            "action_accuracy": control_metrics["action_accuracy"],
        },
        "structured_carm": {
            "task_success_revised": structured_metrics["task_success_revised"],
            "action_accuracy": structured_metrics["action_accuracy"],
            "relation_accuracy": structured_metrics["relation_accuracy"],
            "vision_info_accuracy": structured_metrics["vision_info_accuracy"],
            "text_info_accuracy": structured_metrics["text_info_accuracy"],
        },
    }
    (output_dir / "metrics_snapshot.json").write_text(json.dumps(metrics_snapshot, indent=2), encoding="utf-8")

    _build_summary_markdown(
        output_dir,
        agreement_stats=agreement_stats,
        backbone_metrics=backbone_metrics,
        agreement_metrics=agreement_metrics,
        confidence_metrics=confidence_metrics,
        prompt_metrics=prompt_metrics,
        probe_metrics=probe_metrics,
        control_metrics=control_metrics,
        structured_metrics=structured_metrics,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
