#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# Make scripts runnable without requiring editable install when launched from outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.data.io import load_examples
from carm.data.schema import Split
from carm.eval.baselines import (
    BackboneDirectBaseline,
    ProbeOnlyHeuristicBaseline,
    PromptVerificationBaseline,
    TwoPassSelfConsistencyBaseline,
    UncertaintyThresholdAbstainBaseline,
)
from carm.eval.evaluator import evaluate_predictor
from carm.models.registry import create_backbone
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase A baseline evaluators.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from existing per-baseline outputs.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print per-example progress every N examples inside each baseline (0 disables).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to <output_dir>/run.log.",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="all or comma-separated split names (train,val,test_id,test_ood_family,test_ood_severity,test_ood_hard_swap)",
    )
    return parser.parse_args()


def _parse_split_filter(raw: str) -> set[str] | None:
    if raw.strip().lower() == "all":
        return None
    splits = {s.strip() for s in raw.split(",") if s.strip()}
    valid = {s.value for s in Split}
    unknown = sorted(s for s in splits if s not in valid)
    if unknown:
        raise ValueError(f"Unknown split filters: {unknown}. Valid: {sorted(valid)}")
    return splits


def _make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {msg}"
        print(line)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _log


def _compact_metrics(metrics: dict) -> dict:
    keys = [
        "accuracy",
        "action_accuracy",
        "macro_f1_conflict",
        "ece",
        "brier",
        "monotonicity_violation_rate",
    ]
    return {k: metrics[k] for k in keys if k in metrics}


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    examples = load_examples(args.input_jsonl)

    split_filter = _parse_split_filter(args.split)
    if split_filter is not None:
        examples = [ex for ex in examples if ex.split.value in split_filter]

    if not examples:
        raise ValueError("No examples selected for baseline run.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else out_dir / "run.log"
    log = _make_logger(log_path)
    log(
        f"start baselines config={args.config} input={args.input_jsonl} split={args.split} "
        f"resume={args.resume} examples={len(examples)}"
    )

    backbone = create_backbone(cfg.get("backbone", {}))
    eval_cfg = cfg.get("eval", {})

    baselines = [
        BackboneDirectBaseline(backbone),
        PromptVerificationBaseline(backbone),
        UncertaintyThresholdAbstainBaseline(
            backbone,
            entropy_threshold=float(eval_cfg.get("uncertainty_entropy_threshold", 1.9)),
        ),
        TwoPassSelfConsistencyBaseline(backbone),
        ProbeOnlyHeuristicBaseline(backbone),
    ]

    summary_path = out_dir / "summary.json"
    summary: dict[str, dict] = {}
    if args.resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            summary = loaded
            log(f"loaded existing summary with {len(summary)} baseline entries")

    for baseline in baselines:
        sub = out_dir / baseline.name
        metrics_path = sub / "metrics.json"
        preds_path = sub / "per_example_predictions.jsonl"
        if args.resume and metrics_path.exists() and preds_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            summary[baseline.name] = metrics
            log(f"skip baseline={baseline.name} (metrics already present)")
            continue

        log(f"run baseline={baseline.name}")
        metrics = evaluate_predictor(
            baseline,
            examples,
            output_dir=sub,
            resume=args.resume,
            progress_every=int(args.progress_every),
        )
        summary[baseline.name] = metrics
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"done baseline={baseline.name}")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log("completed all baselines")
    log(f"full summary json: {summary_path}")

    compact = {name: _compact_metrics(metrics) for name, metrics in summary.items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
