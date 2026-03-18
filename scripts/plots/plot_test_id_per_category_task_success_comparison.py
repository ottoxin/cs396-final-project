#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "RUN-0013_hf5way_qwen_test_id_tuned_caption_derived_refresh"
    / "summary.json"
)
DEFAULT_CARM_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "carm"
    / "RUN-0011_hf5way_qwen_caption_derived_eval_test_id"
    / "metrics.json"
)
DEFAULT_OUTPUT_STEM = "test_id_per_category_task_success_carm_vs_baselines"
SKILL_SCRIPTS_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization" / "scripts"
MODEL_ORDER = ["CARM", "agreement_check", "confidence_threshold", "backbone_direct"]
MODEL_COLORS = {
    "CARM": "#C84C0C",
    "agreement_check": "#0E8A6A",
    "confidence_threshold": "#D9A404",
    "backbone_direct": "#355070",
}
CATEGORIES = ["C1", "C2", "C3", "C4", "C5"]
HIGHLIGHT_CATEGORIES = {"C2", "C5"}


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
    parser = argparse.ArgumentParser(
        description="Plot per-category task success on test_id for CARM and selected baselines."
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
        help="Path to the baseline summary.json report.",
    )
    parser.add_argument(
        "--carm-metrics",
        type=Path,
        default=DEFAULT_CARM_METRICS,
        help="Path to the CARM metrics.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output base path without extension. PNG is always produced.",
    )
    return parser.parse_args()


def _default_output_base(baseline_summary: Path) -> Path:
    run_root = baseline_summary.resolve().parent
    return run_root / "report" / DEFAULT_OUTPUT_STEM


def load_baseline_scores(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        summary = json.load(handle)

    scores: dict[str, dict[str, float]] = {}
    for model, payload in summary.items():
        if model not in MODEL_ORDER:
            continue
        category_scores = payload.get("task_success_per_category", {})
        scores[model] = {category: float(category_scores[category]) for category in CATEGORIES}
    return scores


def load_carm_scores(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    category_scores = metrics["task_success_per_category"]
    return {category: float(category_scores[category]) for category in CATEGORIES}


def build_plot(model_scores: dict[str, dict[str, float]]) -> plt.Figure:
    apply_publication_style, set_color_palette, _ = _load_visualization_helpers()
    apply_publication_style("default")
    set_color_palette("okabe_ito")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(CATEGORIES))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(MODEL_ORDER))
    carm_offset = float(offsets[MODEL_ORDER.index("CARM")])

    for offset, model in zip(offsets, MODEL_ORDER, strict=True):
        scores = [model_scores[model][category] for category in CATEGORIES]
        bars = ax.bar(
            x + offset,
            scores,
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for bar, score in zip(bars, scores, strict=True):
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                score + 0.018,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )

    # Emphasize the conflict-heavy categories where CARM is intended to help most.
    for idx, category in enumerate(CATEGORIES):
        if category not in HIGHLIGHT_CATEGORIES:
            continue
        center = x[idx] + carm_offset
        ax.add_patch(
            Rectangle(
                (center - (width * 0.95), -0.02),
                width * 1.9,
                1.01,
                fill=False,
                edgecolor="#FF2B2B",
                linewidth=1.1,
                zorder=4,
                clip_on=False,
            )
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("Task Success")
    ax.set_xlabel("Protocol Category")
    ax.set_title("Per-Category Task Success on test_id: CARM vs Baselines")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    return fig


def main() -> int:
    args = parse_args()
    baseline_summary = args.baseline_summary.resolve()
    carm_metrics = args.carm_metrics.resolve()
    output_base = args.output.resolve() if args.output is not None else _default_output_base(baseline_summary)

    if not baseline_summary.exists():
        raise FileNotFoundError(f"Baseline summary not found: {baseline_summary}")
    if not carm_metrics.exists():
        raise FileNotFoundError(f"CARM metrics not found: {carm_metrics}")

    model_scores = load_baseline_scores(baseline_summary)
    model_scores["CARM"] = load_carm_scores(carm_metrics)

    missing = [model for model in MODEL_ORDER if model not in model_scores]
    if missing:
        raise ValueError(f"Missing model scores for: {missing}")

    fig = build_plot(model_scores)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    _, _, save_publication_figure = _load_visualization_helpers()
    save_publication_figure(fig, output_base, formats=["png"], dpi=300)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
