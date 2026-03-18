#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "RUN-0013_hf5way_qwen_test_id_tuned_caption_derived_refresh"
    / "summary.json"
)
DEFAULT_OUTPUT_BASE = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "RUN-0013_hf5way_qwen_test_id_tuned_caption_derived_refresh"
    / "report"
    / "per_category_accuracy_baselines_refresh"
)
SKILL_SCRIPTS_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization" / "scripts"
MODEL_ORDER = ["agreement_check", "confidence_threshold", "backbone_direct", "probe_heuristic"]
MODEL_COLORS = {
    "agreement_check": "#0E8A6A",
    "confidence_threshold": "#D9A404",
    "backbone_direct": "#355070",
    "probe_heuristic": "#7B61A8",
}
CATEGORIES = ["C1", "C2", "C3", "C4", "C5"]


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
        description="Plot refreshed baseline per-category accuracy on test_id."
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
        help="Path to the refreshed baseline summary.json report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Output base path without extension. PNG is always produced.",
    )
    return parser.parse_args()


def load_accuracy_scores(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    scores: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        payload = summary.get(model)
        if payload is None:
            continue
        category_scores = payload.get("accuracy_per_category", {})
        scores[model] = {category: float(category_scores[category]) for category in CATEGORIES}
    return scores


def build_plot(model_scores: dict[str, dict[str, float]]) -> plt.Figure:
    apply_publication_style, set_color_palette, _ = _load_visualization_helpers()
    apply_publication_style("default")
    set_color_palette("okabe_ito")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(CATEGORIES))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(MODEL_ORDER))

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

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Protocol Category")
    ax.set_title("Per-Category Accuracy on test_id: Refreshed Baselines")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    return fig


def main() -> int:
    args = parse_args()
    baseline_summary = args.baseline_summary.resolve()
    output_base = args.output.resolve()

    if not baseline_summary.exists():
        raise FileNotFoundError(f"Baseline summary not found: {baseline_summary}")

    model_scores = load_accuracy_scores(baseline_summary)
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
