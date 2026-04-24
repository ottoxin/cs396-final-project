#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from carm.data.answer_vocab import canonicalization_mapping_from_family_vocabs, load_family_vocabs
from carm.data.io import load_examples, read_jsonl
from carm.data.schema import ConflictExample, Split
from carm.eval.baselines import (
    AgreementCheckBaseline,
    BackboneDirectBaseline,
    ConfidenceThresholdBaseline,
    ProbeHeuristicBaseline,
)
from carm.experimental.baselines import PROMPT_ONLY_ABSTAIN_ACTION_TEMPLATE, PromptOnlyAbstainBaseline
from carm.experimental.evaluation import (
    StructuredCARMPredictor,
    evaluate_predictor_experimental,
    summarize_feature_diagnostics,
    write_failure_diagnostics,
)
from carm.experimental.labels import derive_labels
from carm.experimental.model import (
    CascadeCARMConfig,
    CascadeCARMHeads,
    DistributionCARMConfig,
    DistributionCARMHeads,
    ExperimentalCARMConfig,
    ExperimentalCARMHeads,
    FlatHiddenCARMConfig,
    FlatHiddenCARMHeads,
)
from carm.experimental.sampling import SmallRunConfig, build_small_run_splits
from carm.experimental.training import ExperimentalLossConfig, ExperimentalTrainer, ExperimentalTrainerConfig
from carm.models.registry import create_backbone
from carm.utils.config import load_yaml_config
from carm.utils.device import resolve_carm_device
from carm.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the revised small-data experimental CARM path.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def _resolve_answer_canonicalization(eval_cfg: dict[str, object], backbone_cfg: dict[str, object]) -> dict[str, object]:
    resolved = dict(eval_cfg.get("answer_canonicalization", {}) or {})
    vocab_path = backbone_cfg.get("family_vocab_path")
    if isinstance(vocab_path, str) and vocab_path.strip():
        resolved.update(canonicalization_mapping_from_family_vocabs(load_family_vocabs(vocab_path)))
    return resolved


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _write_exception_trace(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(traceback.format_exc(), encoding="utf-8")
    return str(path)


def _cfg_value(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _build_schema_audit(
    examples: list[ConflictExample],
    derived_by_id: dict[str, Any],
    *,
    output_path: Path,
) -> dict[str, Any]:
    status_counts = Counter()
    vision_info_counts = Counter()
    text_info_counts = Counter()
    relation_counts = Counter()
    action_counts = Counter()
    image_states = Counter()
    caption_states = Counter()
    split_base_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test_id": set()}
    metric_mismatch = 0

    for example in examples:
        derived = derived_by_id[example.example_id]
        status_counts[derived.derivation_status] += 1
        if derived.vision_info_state:
            vision_info_counts[derived.vision_info_state] += 1
        if derived.text_info_state:
            text_info_counts[derived.text_info_state] += 1
        if derived.pairwise_relation:
            relation_counts[derived.pairwise_relation] += 1
        if derived.action_target:
            action_counts[derived.action_target] += 1
        if derived.metric_semantics_mismatch:
            metric_mismatch += 1
        if isinstance(example.metadata, dict):
            image_states[str(example.metadata.get("image_state", ""))] += 1
            caption_states[str(example.metadata.get("caption_state", ""))] += 1
        if example.split.value in split_base_ids:
            split_base_ids[example.split.value].add(example.base_id)

    overlap_train_val = len(split_base_ids["train"].intersection(split_base_ids["val"]))
    overlap_train_test = len(split_base_ids["train"].intersection(split_base_ids["test_id"]))
    overlap_val_test = len(split_base_ids["val"].intersection(split_base_ids["test_id"]))

    summary = {
        "total_examples": len(examples),
        "derivation_status_counts": dict(sorted(status_counts.items())),
        "observed_vision_info_state_counts": dict(sorted(vision_info_counts.items())),
        "observed_text_info_state_counts": dict(sorted(text_info_counts.items())),
        "observed_pairwise_relation_counts": dict(sorted(relation_counts.items())),
        "derived_action_target_counts": dict(sorted(action_counts.items())),
        "image_state_counts": dict(sorted(image_states.items())),
        "caption_state_counts": dict(sorted(caption_states.items())),
        "metric_semantics_mismatch_examples": int(metric_mismatch),
        "observed_joint_state_count": len({(d.vision_info_state, d.text_info_state, d.pairwise_relation) for d in derived_by_id.values()}),
        "full_three_by_three_grid_supported": False,
        "unsupported_grid_reason": (
            "The benchmark intentionally uses the reduced C1-C5 design rather than a full unrestricted modality-state cross-product."
        ),
        "split_base_overlap": {
            "train_val": overlap_train_val,
            "train_test_id": overlap_train_test,
            "val_test_id": overlap_val_test,
        },
        "template_id_available": any(example.template_id for example in examples),
    }

    lines = [
        "# Data and Schema Audit Summary",
        "",
        f"- total_examples: {summary['total_examples']}",
        f"- derivation_status_counts: {json.dumps(summary['derivation_status_counts'], sort_keys=True)}",
        f"- observed_vision_info_state_counts: {json.dumps(summary['observed_vision_info_state_counts'], sort_keys=True)}",
        f"- observed_text_info_state_counts: {json.dumps(summary['observed_text_info_state_counts'], sort_keys=True)}",
        f"- observed_pairwise_relation_counts: {json.dumps(summary['observed_pairwise_relation_counts'], sort_keys=True)}",
        f"- derived_action_target_counts: {json.dumps(summary['derived_action_target_counts'], sort_keys=True)}",
        f"- metric_semantics_mismatch_examples: {summary['metric_semantics_mismatch_examples']}",
        "",
        "## Risks Checked",
        "",
        "- Label derivability risk: revised supervision depends on the prepared explicit fields rather than the raw HF rows alone, especially for C4 text-supported targets.",
        "- Action-semantics risk: the explicit label contract now treats C4 as both informative plus contradictory with joint abstain; any stale older-head artifact is invalid for this stage.",
        "- Feature adequacy risk: the current feature vector is reused, so any failure to separate contradiction from consistency is still informative about representation limits and is summarized in `feature_diagnostics_summary.md`.",
        "- Metric compatibility risk: legacy task success still exists for comparison, but the stage target is the revised selective metric suite.",
        "- Counterfactual-loss compatibility risk: exposed in config but intentionally disabled for the explicit four-head model.",
        f"- Split leakage risk: base_id overlap train/val={overlap_train_val}, train/test_id={overlap_train_test}, val/test_id={overlap_val_test}.",
        (
            "- Prompt-baseline comparability risk: the action-prompt baseline is implemented against the same backbone interface, "
            "but real-model comparability still depends on running a real multimodal backbone."
        ),
        "",
        "## Unsupported Assumptions",
        "",
        f"- full_three_by_three_grid_supported: {summary['full_three_by_three_grid_supported']}",
        f"- unsupported_grid_reason: {summary['unsupported_grid_reason']}",
        f"- template_id_available: {summary['template_id_available']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _build_baseline_table(
    baseline_results: dict[str, dict[str, Any]],
    *,
    output_csv: Path,
    output_md: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for name, payload in baseline_results.items():
        if payload.get("status") != "ok":
            notes = payload.get("reason")
            if payload.get("traceback_path"):
                notes = f"{notes} (trace: {payload['traceback_path']})"
            rows.append(
                {
                    "panel": "B" if name != "backbone_direct" else "A",
                    "predictor": name,
                    "status": payload.get("status"),
                    "answer_accuracy": None,
                    "coverage": None,
                    "accuracy_on_answered": None,
                    "selective_accuracy": None,
                    "task_success_revised": None,
                    "action_accuracy": None,
                    "risk_coverage_points": None,
                    "contradiction_error_rate": None,
                    "irrelevance_error_rate": None,
                    "notes": notes,
                }
            )
            continue

        metrics = payload["metrics"]
        notes = ""
        panel = "A" if name == "backbone_direct" else "B"
        if name == "backbone_direct":
            notes = "coverage fixed at 1.0; abstention unavailable; abstention-sensitive metrics are not directly comparable"
        rows.append(
            {
                "panel": panel,
                "predictor": name,
                "status": "ok",
                "answer_accuracy": metrics.get("answer_accuracy"),
                "coverage": metrics.get("coverage"),
                "accuracy_on_answered": metrics.get("accuracy_on_answered"),
                "selective_accuracy": metrics.get("selective_accuracy"),
                "task_success_revised": metrics.get("task_success_revised"),
                "action_accuracy": metrics.get("action_accuracy"),
                "risk_coverage_points": len(metrics.get("risk_coverage_task_success_revised", []) or []),
                "contradiction_error_rate": metrics.get("contradiction_error_rate"),
                "irrelevance_error_rate": metrics.get("irrelevance_error_rate"),
                "notes": notes,
            }
        )

    _write_csv(output_csv, rows)
    md_lines = [
        "| panel | predictor | status | answer_accuracy | coverage | accuracy_on_answered | selective_accuracy | task_success_revised | action_accuracy | risk_coverage_points | contradiction_error_rate | irrelevance_error_rate | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {panel} | {predictor} | {status} | {answer_accuracy} | {coverage} | {accuracy_on_answered} | {selective_accuracy} | {task_success_revised} | {action_accuracy} | {risk_coverage_points} | {contradiction_error_rate} | {irrelevance_error_rate} | {notes} |".format(
                **row
            )
        )
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _write_failure_report(
    output_path: Path,
    *,
    run_name: str,
    backbone_name: str,
    schema_summary: dict[str, Any],
    baseline_results: dict[str, dict[str, Any]],
    structured_status: dict[str, Any],
) -> None:
    worked: list[str] = [
        f"Centralized label derivation ran on {schema_summary['total_examples']} prepared examples and wrote a full audit.",
        "Small-run sampling wrote explicit split manifests and selected example IDs.",
    ]
    partial: list[str] = []
    failed: list[str] = []
    unsupported: list[str] = [
        schema_summary["unsupported_grid_reason"],
    ]
    next_fixes: list[str] = [
        "Inspect whether the current feature vector separates C1 vs C4 once the explicit four-head model finishes on the corrected 10 percent subset.",
        "Restore template-family identifiers if split leakage by construction family needs to be audited rigorously.",
    ]
    if "debug" in backbone_name:
        next_fixes.insert(1, "Add a real multimodal GPU-backed run with the same script/config once hardware is available.")

    for name, payload in baseline_results.items():
        status = payload.get("status")
        if status == "ok":
            worked.append(f"{name} executed on the sampled test split.")
        elif status == "unsupported":
            detail = f"{name} is implemented but was unsupported in this environment: {payload.get('reason')}"
            if payload.get("traceback_path"):
                detail += f" (trace: {payload['traceback_path']})"
            partial.append(detail)
        else:
            failed.append(f"{name} failed: {payload.get('reason')}")

    if structured_status.get("status") == "ok":
        worked.append("The structured experimental CARM path trained and evaluated on the sampled train/val/test splits.")
    else:
        detail = f"Structured CARM experimental path failed: {structured_status.get('reason')}"
        if structured_status.get("traceback_path"):
            detail += f" (trace: {structured_status['traceback_path']})"
        failed.append(detail)

    lines = [
        f"# Final Failure Report: {run_name}",
        "",
        "## Worked",
        "",
        *[f"- {item}" for item in worked],
        "",
        "## Partial",
        "",
        *([f"- {item}" for item in partial] or ["- none"]),
        "",
        "## Failed",
        "",
        *([f"- {item}" for item in failed] or ["- none"]),
        "",
        "## Unsupported Assumptions",
        "",
        *[f"- {item}" for item in unsupported],
        "",
        "## Minimal Next Fixes",
        "",
        *[f"- {item}" for item in next_fixes],
        "",
        "## Interpretation",
        "",
        "- backbone_direct answers: this panel shows what happens when the backbone is forced to answer on every example.",
        "- selective baselines answer: these panels show what changes when the system is allowed to abstain or arbitrate.",
        "- structured CARM answers: this panel shows whether explicit structured supervision improves withholding and arbitration under the reduced defensible label space.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    experimental_cfg = dict(cfg.get("experimental", {}) or {})
    seed = int(experimental_cfg.get("random_seed", cfg.get("seed", 7)))
    set_global_seed(seed)

    input_jsonl = (
        Path(args.input_jsonl)
        if args.input_jsonl
        else PROJECT_ROOT / str(_cfg_value(cfg, "data", "paths", "prepared_jsonl", default="data/cache/hf_5way/prepared/carm_vqa_5way.jsonl"))
    )
    run_name = str(experimental_cfg.get("run_name", "RUN-EXP-SMALL"))
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / str(experimental_cfg.get("output_root", "outputs/experimental")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = output_dir / "run_config.json"
    prompt_template_path = output_dir / "prompt_templates" / "prompt_only_abstain_action.txt"
    label_audit_jsonl = output_dir / "label_derivation_audit.jsonl"
    label_audit_csv = output_dir / "label_derivation_audit.csv"
    schema_summary_path = output_dir / "data_schema_audit_summary.md"
    split_manifest_path = output_dir / "split_manifest.json"
    split_stats_path = output_dir / "split_stats.json"
    failure_report_path = output_dir / "failure_report.md"
    comparison_csv = output_dir / "baseline_comparison.csv"
    comparison_md = output_dir / "baseline_comparison.md"

    examples = load_examples(input_jsonl)
    derived_by_id = {example.example_id: derive_labels(example) for example in examples}
    label_rows = [derived.to_audit_row() for derived in derived_by_id.values()]
    _write_jsonl(label_audit_jsonl, label_rows)
    _write_csv(label_audit_csv, label_rows)

    schema_summary = _build_schema_audit(examples, derived_by_id, output_path=schema_summary_path)

    small_run_cfg = SmallRunConfig(
        max_train_examples=int(experimental_cfg.get("max_train_examples", 48)),
        max_val_examples=int(experimental_cfg.get("max_val_examples", 24)),
        max_test_examples=int(experimental_cfg.get("max_test_examples", 24)),
        random_seed=seed,
        sampling_strategy=str(experimental_cfg.get("sampling_strategy", "protocol_category_family")),
        run_name=run_name,
    )
    selected_splits, split_manifest = build_small_run_splits(examples, derived_by_id, small_run_cfg)
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    split_stats = {
        split: {
            "count": len(rows),
            "example_ids": [example.example_id for example in rows],
        }
        for split, rows in selected_splits.items()
    }
    split_stats_path.write_text(json.dumps(split_stats, indent=2), encoding="utf-8")

    prompt_template_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_template_path.write_text(PROMPT_ONLY_ABSTAIN_ACTION_TEMPLATE, encoding="utf-8")

    backbone = create_backbone(cfg.get("backbone", {}))
    canonicalization_cfg = _resolve_answer_canonicalization(cfg.get("eval", {}), cfg.get("backbone", {}))
    baseline_results: dict[str, dict[str, Any]] = {}

    baseline_specs = {
        "backbone_direct": BackboneDirectBaseline(backbone),
        "agreement_check": AgreementCheckBaseline(backbone),
        "confidence_threshold": ConfidenceThresholdBaseline(
            backbone,
            threshold=float(_cfg_value(cfg, "eval", "confidence_threshold", default=0.3)),
        ),
        "probe_heuristic": ProbeHeuristicBaseline(
            backbone,
            both_uncertain_threshold=float(_cfg_value(cfg, "eval", "probe_both_uncertain_threshold", default=2.0)),
        ),
        "prompt_only_abstain": PromptOnlyAbstainBaseline(backbone),
    }

    sampled_test = list(selected_splits.get(Split.TEST_ID.value, []))
    for name, predictor in baseline_specs.items():
        baseline_dir = output_dir / "baselines" / name
        try:
            metrics = evaluate_predictor_experimental(
                predictor,
                backbone,
                sampled_test,
                derived_by_id,
                output_dir=baseline_dir,
                canonicalization_cfg=canonicalization_cfg,
            )
            baseline_results[name] = {"status": "ok", "metrics": metrics, "output_dir": str(baseline_dir)}
        except Exception as exc:
            baseline_results[name] = {
                "status": "unsupported",
                "reason": str(exc),
                "traceback_path": _write_exception_trace(baseline_dir / "exception_traceback.txt"),
                "output_dir": str(baseline_dir),
            }

    model_cfg = cfg.get("model", {})
    training_cfg = dict(cfg.get("training", {}) or {})
    experimental_training_cfg = dict(experimental_cfg.get("training", {}) or {})
    experimental_loss_cfg = dict(experimental_cfg.get("loss", {}) or {})
    device = resolve_carm_device(training_cfg.get("device"), backbone)
    structured_dir = output_dir / "structured_carm"

    model_type = str(model_cfg.get("type", "flat")).strip().lower()
    if model_type == "flat_hidden":
        model: ExperimentalCARMHeads | CascadeCARMHeads | DistributionCARMHeads | FlatHiddenCARMHeads = FlatHiddenCARMHeads(
            FlatHiddenCARMConfig(
                hidden_size=int(model_cfg.get("hidden_size", 128)),
                probe_feature_size=int(model_cfg.get("probe_feature_size", 3)),
                cross_modal_feature_size=int(model_cfg.get("cross_modal_feature_size", 5)),
                pool=str(model_cfg.get("pool", "mean")),
                trunk_hidden_size=model_cfg.get("trunk_hidden_size"),
            )
        )
    elif model_type == "distribution":
        model: ExperimentalCARMHeads | CascadeCARMHeads | DistributionCARMHeads = DistributionCARMHeads(
            DistributionCARMConfig(
                vocab_size=int(model_cfg.get("vocab_size", 35)),
                cross_modal_feature_size=int(model_cfg.get("cross_modal_feature_size", 5)),
                trunk_hidden_size=int(model_cfg.get("trunk_hidden_size", 128)),
                stage_hidden_size=int(model_cfg.get("stage_hidden_size", 64)),
            )
        )
    elif model_type == "cascade":
        model = CascadeCARMHeads(
            CascadeCARMConfig(
                hidden_size=int(model_cfg.get("hidden_size", 128)),
                probe_feature_size=int(model_cfg.get("probe_feature_size", 3)),
                cross_modal_feature_size=int(model_cfg.get("cross_modal_feature_size", 5)),
                pool=str(model_cfg.get("pool", "mean")),
                trunk_hidden_size=model_cfg.get("trunk_hidden_size"),
                stage_hidden_size=int(model_cfg.get("stage_hidden_size", 64)),
            )
        )
    else:
        model = ExperimentalCARMHeads(
            ExperimentalCARMConfig(
                hidden_size=int(model_cfg.get("hidden_size", 128)),
                probe_feature_size=int(model_cfg.get("probe_feature_size", 3)),
                pool=str(model_cfg.get("pool", "mean")),
            )
        )

    def _evaluate_structured(current_model: ExperimentalCARMHeads | CascadeCARMHeads | DistributionCARMHeads, split_examples: list[ConflictExample], out_dir: Path) -> dict[str, Any]:
        predictor = StructuredCARMPredictor(current_model, backbone, device=device)
        return evaluate_predictor_experimental(
            predictor,
            backbone,
            split_examples,
            derived_by_id,
            output_dir=out_dir,
            canonicalization_cfg=canonicalization_cfg,
        )

    trainer = ExperimentalTrainer(
        model=model,
        backbone=backbone,
        derived_by_id=derived_by_id,
        config=ExperimentalTrainerConfig(
            batch_size=int(experimental_training_cfg.get("batch_size", training_cfg.get("batch_size", 4))),
            epochs=int(experimental_training_cfg.get("epochs", training_cfg.get("epochs", 2))),
            lr=float(experimental_training_cfg.get("lr", training_cfg.get("lr", 1e-3))),
            weight_decay=float(experimental_training_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.01))),
            patience=int(experimental_training_cfg.get("patience", training_cfg.get("patience", 2))),
            device=device,
            early_stop_metric=str(experimental_training_cfg.get("early_stop_metric", "task_success_revised")),
            log_every_steps=int(experimental_training_cfg.get("log_every_steps", training_cfg.get("log_every_steps", 50))),
            loss=ExperimentalLossConfig(
                lambda_vision_info=float(experimental_loss_cfg.get("lambda_vision_info", experimental_loss_cfg.get("lambda_info", 1.0))),
                lambda_text_info=float(experimental_loss_cfg.get("lambda_text_info", experimental_loss_cfg.get("lambda_info", 1.0))),
                lambda_relation=float(experimental_loss_cfg.get("lambda_relation", experimental_loss_cfg.get("lambda_rel", 1.0))),
                lambda_action=float(experimental_loss_cfg.get("lambda_action", 1.0)),
                lambda_cf=float(experimental_loss_cfg.get("lambda_cf", 0.0)),
                margin_cf=float(experimental_loss_cfg.get("margin_cf", 0.2)),
                action_class_weights=[float(w) for w in experimental_loss_cfg["action_class_weights"]] if "action_class_weights" in experimental_loss_cfg else None,
            ),
        ),
        evaluate_fn=_evaluate_structured,
    )

    structured_status: dict[str, Any] = {"status": "failed", "reason": "not_run"}
    try:
        train_examples = list(selected_splits.get(Split.TRAIN.value, []))
        val_examples = list(selected_splits.get(Split.VAL.value, []))
        test_examples = list(selected_splits.get(Split.TEST_ID.value, []))
        training_result = trainer.train(train_examples, val_examples, output_dir=structured_dir / "train")

        ckpt_payload = {
            "model_state_dict": training_result.best_model_state_dict,
            "config": {
                "resolved_device": device,
                "experimental": experimental_cfg,
                "loss": trainer.config.loss.to_dict(),
            },
            "best_val_metrics": training_result.best_val_metrics,
            "label_availability": training_result.label_availability,
        }
        torch.save(ckpt_payload, structured_dir / "train" / "structured_carm_heads.pt")
        test_metrics = _evaluate_structured(model, test_examples, structured_dir / "test")
        structured_status = {
            "status": "ok",
            "reason": None,
            "best_val_metrics": training_result.best_val_metrics,
            "test_metrics": test_metrics,
        }

        predictions_rows = _load_predictions(structured_dir / "test" / "per_example_predictions.jsonl")
        write_failure_diagnostics(predictions_rows, structured_dir / "test" / "failure_diagnostics.csv")
        summarize_feature_diagnostics(predictions_rows, structured_dir / "test" / "feature_diagnostics_summary.md")
    except Exception as exc:
        structured_status = {
            "status": "failed",
            "reason": str(exc),
            "traceback_path": _write_exception_trace(structured_dir / "failure_traceback.txt"),
        }

    run_config = {
        "config_path": args.config,
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "run_name": run_name,
        "seed": seed,
        "resolved_device": device,
        "experimental_cfg": experimental_cfg,
        "backbone_name": getattr(backbone, "name", type(backbone).__name__),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    comparison_dir = output_dir / "report"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    all_comparison_results = dict(baseline_results)
    if structured_status.get("status") == "ok":
        all_comparison_results["structured_carm_experimental"] = {
            "status": "ok",
            "metrics": structured_status["test_metrics"],
            "output_dir": str(structured_dir / "test"),
        }
    else:
        all_comparison_results["structured_carm_experimental"] = {
            "status": "failed",
            "reason": structured_status.get("reason"),
            "output_dir": str(structured_dir / "test"),
        }
    _build_baseline_table(all_comparison_results, output_csv=comparison_csv, output_md=comparison_md)

    _write_failure_report(
        failure_report_path,
        run_name=run_name,
        backbone_name=str(getattr(backbone, "name", type(backbone).__name__)),
        schema_summary=schema_summary,
        baseline_results=baseline_results,
        structured_status=structured_status,
    )

    print(json.dumps(
        {
            "run_name": run_name,
            "output_dir": str(output_dir),
            "baselines": {name: payload.get("status") for name, payload in baseline_results.items()},
            "structured_status": structured_status.get("status"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
