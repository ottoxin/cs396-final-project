#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "RUN-0010_hf5way_qwen_test_id_tuned_caption_derived"
    / "backbone_direct"
    / "metrics.json"
)
SKILL_SCRIPTS_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization" / "scripts"
PLOT_SPECS = {
    "task_success": {
        "category_key": "task_success_per_category",
        "overall_key": "task_success",
        "ylabel": "Task Success",
        "title": "Backbone Direct Baseline: Per-Category Task Success on test_id",
        "output_stem": "backbone_direct_per_category_task_success_test_id",
        "bar_color": "#E69F00",
        "edge_color": "#A05A00",
    },
    "accuracy": {
        "category_key": "accuracy_per_category",
        "overall_key": "accuracy",
        "ylabel": "Accuracy",
        "title": "Backbone Direct Baseline: Per-Category Accuracy on test_id",
        "output_stem": "backbone_direct_per_category_accuracy_test_id",
        "bar_color": "#56B4E9",
        "edge_color": "#0072B2",
    },
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
    parser = argparse.ArgumentParser(
        description="Plot per-category metrics for the backbone_direct baseline on test_id."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the backbone_direct metrics.json file.",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(PLOT_SPECS.keys()),
        default="task_success",
        help="Which per-category metric to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output base path without extension. PNG is always produced.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_output_base(metrics_path: Path, metric_name: str) -> Path:
    baseline_dir = metrics_path.resolve().parent
    run_root = baseline_dir.parent
    return run_root / "report" / str(PLOT_SPECS[metric_name]["output_stem"])


def build_plot(metrics: dict, metric_name: str) -> plt.Figure:
    spec = PLOT_SPECS[metric_name]
    category_scores = metrics[str(spec["category_key"])]
    categories = ["C1", "C2", "C3", "C4", "C5"]
    scores = [float(category_scores[category]) for category in categories]
    overall = float(metrics[str(spec["overall_key"])])

    apply_publication_style, set_color_palette, _ = _load_visualization_helpers()
    apply_publication_style("default")
    set_color_palette("okabe_ito")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(categories))
    bar_width = 0.66

    bars = ax.bar(
        x,
        scores,
        width=bar_width,
        color=str(spec["bar_color"]),
        edgecolor=str(spec["edge_color"]),
        linewidth=0.9,
        zorder=2,
    )
    ax.axhline(
        overall,
        color="#999999",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            score + 0.025,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.text(
        len(categories) - 0.45,
        overall + 0.02,
        f"overall = {overall:.3f}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_xlabel("Protocol Category")
    ax.set_title(str(spec["title"]))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    return fig


def main() -> int:
    args = parse_args()
    metrics_path = args.metrics.resolve()
    output_base = (
        args.output.resolve()
        if args.output is not None
        else _default_output_base(metrics_path, args.metric)
    )

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = load_metrics(metrics_path)
    fig = build_plot(metrics, args.metric)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    _, _, save_publication_figure = _load_visualization_helpers()
    save_publication_figure(fig, output_base, formats=["png"], dpi=300)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
