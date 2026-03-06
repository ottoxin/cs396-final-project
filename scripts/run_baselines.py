#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Callable

# Make scripts runnable without requiring editable install when launched from outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.data.io import load_examples
from carm.data.schema import ConflictExample, Split
from carm.eval.baselines import (
    AgreementCheckBaseline,
    BackboneDirectBaseline,
    ConfidenceThresholdBaseline,
    ProbeHeuristicBaseline,
)
from carm.eval.evaluator import evaluate_predictor
from carm.models.registry import create_backbone
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline evaluators.")
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
    parser.add_argument(
        "--track",
        choices=["answer", "policy", "all"],
        default="all",
        help="Accepted for CLI compatibility; flattened evaluator outputs are track-agnostic.",
    )
    parser.add_argument(
        "--schema-version",
        default="2.0",
        help="Accepted for CLI compatibility with older runs.",
    )
    parser.add_argument(
        "--answer-strategy",
        choices=["open_canonicalize"],
        default="open_canonicalize",
        help="Accepted for CLI compatibility with older runs.",
    )
    parser.add_argument(
        "--report-calibration-heuristic",
        action="store_true",
        help="Accepted for CLI compatibility with older runs.",
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


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{max(0.0, seconds):.1f}s"

    total_seconds = max(0, int(seconds))
    mins, secs = divmod(total_seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:d}h{mins:02d}m{secs:02d}s"
    return f"{mins:d}m{secs:02d}s"


def _resolve_image_path(path_value: str, *, project_root: Path = PROJECT_ROOT) -> Path | None:
    direct = Path(path_value)
    if direct.exists():
        return direct.resolve()

    rooted = (project_root / path_value).resolve()
    if rooted.exists():
        return rooted

    return None


def _resolve_example_image_paths(
    examples: list[ConflictExample],
    log: Callable[[str], None],
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    missing: list[tuple[str, str]] = []
    normalized = 0

    for ex in examples:
        resolved = _resolve_image_path(ex.image_path, project_root=project_root)
        if resolved is None:
            missing.append((ex.example_id, ex.image_path))
            continue

        resolved_str = str(resolved)
        if ex.image_path != resolved_str:
            ex.image_path = resolved_str
            normalized += 1

    if missing:
        preview = "; ".join(f"{ex_id} -> {path}" for ex_id, path in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (showing 5/{len(missing)})"
        raise FileNotFoundError(f"Missing image_path for {len(missing)} examples{suffix}: {preview}")

    log(f"validated image paths for {len(examples)} examples; normalized={normalized}")


def _make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {msg}"
        print(line)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _log


def _compact_metrics(entry: dict[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for key in [
        "task_success",
        "accuracy",
        "coverage",
        "accuracy_on_answered",
    ]:
        if key in entry:
            compact[key] = entry[key]
    return compact


def _prune_summary(summary: dict[str, dict], active_names: set[str]) -> tuple[dict[str, dict], list[str]]:
    stale = sorted(k for k in summary if k not in active_names)
    if not stale:
        return summary, []
    pruned = {k: v for k, v in summary.items() if k in active_names}
    return pruned, stale


def main() -> None:
    run_start = time.monotonic()
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
        f"resume={args.resume} track={args.track} schema_version={args.schema_version} examples={len(examples)}"
    )
    _resolve_example_image_paths(examples, log)

    backbone = create_backbone(cfg.get("backbone", {}))
    eval_cfg = cfg.get("eval", {})

    baselines = [
        BackboneDirectBaseline(backbone),
        AgreementCheckBaseline(backbone),
        ConfidenceThresholdBaseline(
            backbone,
            threshold=float(eval_cfg.get("confidence_threshold", 0.3)),
        ),
        ProbeHeuristicBaseline(
            backbone,
            both_uncertain_threshold=float(eval_cfg.get("probe_both_uncertain_threshold", 2.0)),
        ),
    ]
    active_baseline_names = {b.name for b in baselines}

    summary_path = out_dir / "summary.json"
    summary: dict[str, dict] = {}
    if args.resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            summary = loaded
            summary, stale = _prune_summary(summary, active_baseline_names)
            log(f"loaded existing summary with {len(summary)} baseline entries")
            if stale:
                log(f"pruned stale baseline summary entries: {', '.join(stale)}")

    for baseline in baselines:
        sub = out_dir / baseline.name

        log(f"run baseline={baseline.name}")
        baseline_start = time.monotonic()
        results = evaluate_predictor(
            baseline,
            examples,
            output_dir=sub,
            track=args.track,
            schema_version=args.schema_version,
            resume=args.resume,
            progress_every=int(args.progress_every),
            log_fn=log,
            semantic_match_threshold=float(eval_cfg.get("semantic_match_threshold", 0.82)),
            canonicalization_cfg=eval_cfg.get("answer_canonicalization", {}),
            include_heuristic_calibration=bool(args.report_calibration_heuristic),
        )
        summary[baseline.name] = results
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"done baseline={baseline.name} elapsed={_format_duration(time.monotonic() - baseline_start)}")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"completed baselines elapsed={_format_duration(time.monotonic() - run_start)}")
    log(f"full summary json: {summary_path}")

    compact = {name: _compact_metrics(metrics) for name, metrics in summary.items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
