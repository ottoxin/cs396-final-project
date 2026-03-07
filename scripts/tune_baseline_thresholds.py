#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Make scripts runnable without requiring editable install when launched from outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.data.answer_vocab import canonicalization_mapping_from_family_vocabs, load_family_vocabs
from carm.data.io import load_examples
from carm.data.schema import ConflictExample, Split
from carm.eval.baselines import BaseBaseline
from carm.eval.canonicalization import CanonicalizationConfig, canonicalize_answer
from carm.eval.metrics import task_success_from_components
from carm.models.registry import create_backbone
from carm.utils.config import load_yaml_config
from carm.utils.run_metadata import hash_file_contents, hash_jsonable, resolve_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune frozen baseline thresholds on a validation split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test_id", "test_ood_family", "test_ood_severity", "test_ood_hard_swap"],
    )
    parser.add_argument(
        "--objective",
        default="task_success",
        choices=["task_success"],
        help="Primary optimization objective for frozen threshold selection.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print precompute progress every N examples per thresholded baseline (0 disables).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to <output_dir>/run.log.",
    )
    return parser.parse_args()


def _make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {msg}"
        print(line)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _log


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


def _resolve_answer_canonicalization(eval_cfg: dict[str, object], backbone_cfg: dict[str, object]) -> dict[str, object]:
    resolved = dict(eval_cfg.get("answer_canonicalization", {}) or {})
    vocab_path = backbone_cfg.get("family_vocab_path")
    if isinstance(vocab_path, str) and vocab_path.strip():
        resolved.update(canonicalization_mapping_from_family_vocabs(load_family_vocabs(vocab_path)))
    return resolved


def _resolve_dataset_manifest_hash(cfg: dict[str, object], *, project_root: Path = PROJECT_ROOT) -> str | None:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return None
    paths_cfg = data_cfg.get("paths", {})
    if not isinstance(paths_cfg, dict):
        return None
    manifest_path = paths_cfg.get("prepared_manifest_json")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        return None
    candidate = Path(manifest_path)
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return hash_file_contents(candidate)


def _canonicalization_config(mapping: dict[str, object], semantic_match_threshold: float) -> CanonicalizationConfig:
    del semantic_match_threshold
    return CanonicalizationConfig.from_mapping(mapping)


def _answers_match(pred_text: str, gold_text: str, ex: ConflictExample, cfg: CanonicalizationConfig) -> bool:
    pred = canonicalize_answer(pred_text, ex.answer_type, cfg=cfg)
    gold = canonicalize_answer(gold_text, ex.answer_type, cfg=cfg)
    if pred.canonical_label is not None and gold.canonical_label is not None:
        return pred.canonical_label == gold.canonical_label
    return bool(pred.normalized_text) and pred.normalized_text == gold.normalized_text


def _protocol_category(ex: ConflictExample) -> str:
    if isinstance(ex.metadata, dict):
        return str(ex.metadata.get("protocol_category", "")).strip()
    return ""


def _precompute_outcomes(ex: ConflictExample, would_be_correct: bool) -> tuple[float, float]:
    category = _protocol_category(ex)
    answered_success = task_success_from_components(
        ex.oracle_action.value,
        False,
        would_be_correct,
        protocol_category=category,
    )
    abstained_success = task_success_from_components(
        ex.oracle_action.value,
        True,
        False,
        protocol_category=category,
    )
    return float(answered_success), float(abstained_success)


def _maybe_log_progress(
    *,
    label: str,
    current: int,
    total: int,
    start: float,
    progress_every: int,
    log: Callable[[str], None],
) -> None:
    if progress_every <= 0:
        return
    if current % progress_every != 0 and current != total:
        return
    elapsed = max(1e-6, time.monotonic() - start)
    rate = current / elapsed
    remaining = max(0, total - current)
    eta = _format_duration(remaining / rate) if rate > 0 else "n/a"
    log(f"{label} precompute {current}/{total} elapsed={_format_duration(elapsed)} eta={eta} rate={rate:.2f} ex/s")


def _build_confidence_entries(
    backbone: Any,
    examples: list[ConflictExample],
    canon_cfg: CanonicalizationConfig,
    *,
    progress_every: int,
    log: Callable[[str], None],
) -> list[dict[str, float | str]]:
    entries: list[dict[str, float | str]] = []
    start = time.monotonic()
    total = len(examples)

    for idx, ex in enumerate(examples, start=1):
        vision_payload = BaseBaseline._vision_payload(ex)
        mm = backbone.run_backbone_multimodal(vision_payload, ex.text_input, ex.question)
        confidence = 1.0 - BaseBaseline._normalized_entropy(mm.answer_dist)
        would_be_correct = _answers_match(mm.answer_text, ex.gold_answer, ex, canon_cfg)
        answered_success, abstained_success = _precompute_outcomes(ex, would_be_correct)
        entries.append(
            {
                "confidence": float(confidence),
                "would_be_correct": float(would_be_correct),
                "answered_success": float(answered_success),
                "abstained_success": float(abstained_success),
            }
        )
        _maybe_log_progress(
            label="confidence_threshold",
            current=idx,
            total=total,
            start=start,
            progress_every=progress_every,
            log=log,
        )

    return entries


def _build_probe_entries(
    backbone: Any,
    examples: list[ConflictExample],
    canon_cfg: CanonicalizationConfig,
    *,
    progress_every: int,
    log: Callable[[str], None],
) -> list[dict[str, float | str]]:
    entries: list[dict[str, float | str]] = []
    start = time.monotonic()
    total = len(examples)

    for idx, ex in enumerate(examples, start=1):
        vision_payload = BaseBaseline._vision_payload(ex)
        pv = backbone.run_probe_vision_only(vision_payload, ex.question)
        pt = backbone.run_probe_text_only(ex.text_input, ex.question)
        ent_v = BaseBaseline._entropy(pv.answer_dist)
        ent_t = BaseBaseline._entropy(pt.answer_dist)
        chosen = pv if ent_v <= ent_t else pt
        would_be_correct = _answers_match(chosen.answer_text, ex.gold_answer, ex, canon_cfg)
        answered_success, abstained_success = _precompute_outcomes(ex, would_be_correct)
        entries.append(
            {
                "min_entropy": float(min(ent_v, ent_t)),
                "would_be_correct": float(would_be_correct),
                "answered_success": float(answered_success),
                "abstained_success": float(abstained_success),
            }
        )
        _maybe_log_progress(
            label="probe_heuristic",
            current=idx,
            total=total,
            start=start,
            progress_every=progress_every,
            log=log,
        )

    return entries


def _finalize_metrics(*, total_success: float, answered_count: int, answered_correct: float, total: int) -> dict[str, float]:
    coverage = answered_count / total if total else 0.0
    accuracy = answered_correct / total if total else 0.0
    accuracy_on_answered = answered_correct / answered_count if answered_count else 0.0
    task_success = total_success / total if total else 0.0
    return {
        "task_success": float(task_success),
        "coverage": float(coverage),
        "accuracy": float(accuracy),
        "accuracy_on_answered": float(accuracy_on_answered),
    }


def _sweep_confidence_threshold(entries: list[dict[str, float | str]]) -> list[dict[str, float]]:
    if not entries:
        return []
    eps = 1e-12
    scores = np.asarray([float(entry["confidence"]) for entry in entries], dtype=float)
    correct = np.asarray([float(entry["would_be_correct"]) for entry in entries], dtype=float)
    answered_success = np.asarray([float(entry["answered_success"]) for entry in entries], dtype=float)
    abstained_success = np.asarray([float(entry["abstained_success"]) for entry in entries], dtype=float)

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    sorted_answered_success = answered_success[order]
    sorted_abstained_success = abstained_success[order]
    total = len(entries)

    prefix_abstained_success = np.concatenate(([0.0], np.cumsum(sorted_abstained_success)))
    suffix_answered_success = np.concatenate((np.cumsum(sorted_answered_success[::-1])[::-1], [0.0]))
    suffix_answered_correct = np.concatenate((np.cumsum(sorted_correct[::-1])[::-1], [0.0]))

    thresholds = sorted(set(float(x) for x in scores.tolist()))
    thresholds.append(float(np.max(scores) + eps))

    sweep: list[dict[str, float]] = []
    for threshold in thresholds:
        abstained_count = int(np.searchsorted(sorted_scores, threshold, side="left"))
        answered_count = total - abstained_count
        total_success = float(prefix_abstained_success[abstained_count] + suffix_answered_success[abstained_count])
        answered_correct = float(suffix_answered_correct[abstained_count])
        sweep.append(
            {
                "threshold": float(threshold),
                **_finalize_metrics(
                    total_success=total_success,
                    answered_count=answered_count,
                    answered_correct=answered_correct,
                    total=total,
                ),
            }
        )
    return sweep


def _sweep_probe_threshold(entries: list[dict[str, float | str]]) -> list[dict[str, float]]:
    if not entries:
        return []
    eps = 1e-12
    scores = np.asarray([float(entry["min_entropy"]) for entry in entries], dtype=float)
    correct = np.asarray([float(entry["would_be_correct"]) for entry in entries], dtype=float)
    answered_success = np.asarray([float(entry["answered_success"]) for entry in entries], dtype=float)
    abstained_success = np.asarray([float(entry["abstained_success"]) for entry in entries], dtype=float)

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    sorted_answered_success = answered_success[order]
    sorted_abstained_success = abstained_success[order]
    total = len(entries)

    prefix_answered_success = np.concatenate(([0.0], np.cumsum(sorted_answered_success)))
    prefix_answered_correct = np.concatenate(([0.0], np.cumsum(sorted_correct)))
    suffix_abstained_success = np.concatenate((np.cumsum(sorted_abstained_success[::-1])[::-1], [0.0]))

    thresholds = [float(np.min(scores) - eps)]
    thresholds.extend(sorted(set(float(x) for x in scores.tolist())))

    sweep: list[dict[str, float]] = []
    for threshold in thresholds:
        answered_count = int(np.searchsorted(sorted_scores, threshold, side="right"))
        total_success = float(prefix_answered_success[answered_count] + suffix_abstained_success[answered_count])
        answered_correct = float(prefix_answered_correct[answered_count])
        sweep.append(
            {
                "threshold": float(threshold),
                **_finalize_metrics(
                    total_success=total_success,
                    answered_count=answered_count,
                    answered_correct=answered_correct,
                    total=total,
                ),
            }
        )
    return sweep


def _select_best_candidate(
    candidates: list[dict[str, float]],
    *,
    threshold_key: str,
    prefer_lower_threshold: bool,
) -> dict[str, float]:
    if not candidates:
        raise ValueError("No sweep candidates were generated.")

    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate["task_success"] > best["task_success"] + 1e-12:
            best = candidate
            continue
        if abs(candidate["task_success"] - best["task_success"]) > 1e-12:
            continue
        if candidate["coverage"] > best["coverage"] + 1e-12:
            best = candidate
            continue
        if abs(candidate["coverage"] - best["coverage"]) > 1e-12:
            continue
        if prefer_lower_threshold:
            if candidate[threshold_key] < best[threshold_key] - 1e-12:
                best = candidate
        else:
            if candidate[threshold_key] > best[threshold_key] + 1e-12:
                best = candidate
    return best


def main() -> None:
    run_start = time.monotonic()
    args = parse_args()
    cfg = load_yaml_config(args.config)
    examples = load_examples(args.input_jsonl)
    wanted_split = Split(args.split)
    examples = [ex for ex in examples if ex.split == wanted_split]
    if not examples:
        raise ValueError(f"No examples selected for split={args.split}.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_dir / "run.log"
    log = _make_logger(log_path)
    log(
        f"start threshold tuning config={args.config} input={args.input_jsonl} split={args.split} "
        f"objective={args.objective} examples={len(examples)}"
    )
    _resolve_example_image_paths(examples, log)

    backbone_cfg = cfg.get("backbone", {})
    eval_cfg = cfg.get("eval", {})
    backbone = create_backbone(backbone_cfg)
    canon_mapping = _resolve_answer_canonicalization(eval_cfg, backbone_cfg)
    canon_cfg = _canonicalization_config(
        canon_mapping,
        semantic_match_threshold=float(eval_cfg.get("semantic_match_threshold", 0.82)),
    )

    confidence_entries = _build_confidence_entries(
        backbone,
        examples,
        canon_cfg,
        progress_every=int(args.progress_every),
        log=log,
    )
    probe_entries = _build_probe_entries(
        backbone,
        examples,
        canon_cfg,
        progress_every=int(args.progress_every),
        log=log,
    )

    confidence_sweep = _sweep_confidence_threshold(confidence_entries)
    probe_sweep = _sweep_probe_threshold(probe_entries)
    selected_confidence = _select_best_candidate(
        confidence_sweep,
        threshold_key="threshold",
        prefer_lower_threshold=True,
    )
    selected_probe = _select_best_candidate(
        probe_sweep,
        threshold_key="threshold",
        prefer_lower_threshold=False,
    )

    confidence_sweep_path = out_dir / "confidence_threshold_sweep.json"
    probe_sweep_path = out_dir / "probe_heuristic_sweep.json"
    confidence_sweep_path.write_text(
        json.dumps(
            {
                "baseline": "confidence_threshold",
                "objective": args.objective,
                "split": args.split,
                "candidates": confidence_sweep,
                "selected": selected_confidence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    probe_sweep_path.write_text(
        json.dumps(
            {
                "baseline": "probe_heuristic",
                "objective": args.objective,
                "split": args.split,
                "candidates": probe_sweep,
                "selected": selected_probe,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tuned_payload = {
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config_path": str(Path(args.config)),
        "input_jsonl": str(Path(args.input_jsonl)),
        "selected_split": args.split,
        "objective": args.objective,
        "tie_breakers": {
            "primary": "task_success",
            "secondary": "coverage",
            "confidence_threshold": "lowest_threshold",
            "probe_both_uncertain_threshold": "highest_threshold",
        },
        "thresholds": {
            "confidence_threshold": float(selected_confidence["threshold"]),
            "probe_both_uncertain_threshold": float(selected_probe["threshold"]),
        },
        "selected_val_metrics": {
            "confidence_threshold": {
                key: float(selected_confidence[key])
                for key in ("task_success", "coverage", "accuracy", "accuracy_on_answered")
            },
            "probe_heuristic": {
                key: float(selected_probe[key])
                for key in ("task_success", "coverage", "accuracy", "accuracy_on_answered")
            },
        },
        "resolved_config_hash": hash_jsonable(cfg),
        "dataset_manifest_hash": _resolve_dataset_manifest_hash(cfg),
        "git_commit": resolve_git_commit(PROJECT_ROOT),
        "sweep_artifacts": {
            "confidence_threshold": str(confidence_sweep_path),
            "probe_heuristic": str(probe_sweep_path),
        },
    }

    tuned_path = out_dir / "tuned_thresholds.json"
    tuned_path.write_text(json.dumps(tuned_payload, indent=2), encoding="utf-8")

    log(
        "selected thresholds "
        f"confidence_threshold={selected_confidence['threshold']:.12f} "
        f"probe_both_uncertain_threshold={selected_probe['threshold']:.12f}"
    )
    log(f"completed threshold tuning elapsed={_format_duration(time.monotonic() - run_start)}")
    log(f"frozen thresholds artifact: {tuned_path}")

    print(json.dumps(tuned_payload, indent=2))


if __name__ == "__main__":
    main()
