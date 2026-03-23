# REPORT.md - Phase A Results Ledger

Purpose: canonical experiment results ledger for Phase A. This file is the truth source for run-level outputs and acceptance-gate status.

Last updated: 2026-03-18 21:35 UTC

## Phase Status Snapshot
- current_phase: Phase A
- latest_dataset_manifest_id: data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json
- latest_baseline_summary_artifact: outputs/baselines/RUN-0013_hf5way_qwen_test_id_tuned_caption_derived_refresh/report/main_table.md (provisional; example-count mismatch vs current manifest)
- latest_submission_entrypoint: scripts/submit_experimental_quest.sh; scripts/submit_carm_quest.sh
- latest_experimental_small_run_artifact: outputs/experimental/RUN-EXP-0007_10pct_qwen_protocol/structured_carm/test/metrics.json
- latest_analysis_artifact: outputs/analysis/RUN-ANALYSIS-0001_10pct_protocol/analysis_summary.md
- active_campaign: full_dataset_refresh_for_paper_quality
- quality_bar: top-paper-grade refreshed evidence on the canonical full split

## Active Campaign (2026-03-18)

- Paper target has moved from the archived 10% main table to the canonical full split.
- Established experiment families (`Cascade`, `Flat Hidden`, `Distribution`) are eligible for direct full-data runs.
- New experiment ideas must stay on tiny or small pilots until they show enough signal to justify full-data GPU cost.
- Human-label collection is intentionally skipped in this cycle by user request.
- Any full-data artifact whose example count does not match the current canonical manifest is treated as provisional until rerun.

### Submitted Run Queue

| run_id | job_id | status | notes | artifact_root |
| --- | --- | --- | --- | --- |
| RUN-0014 | 3108359 | queued | full-data val threshold refresh on current canonical manifest | outputs/baselines/RUN-0014_hf5way_qwen_val_tuning_refresh_current |
| RUN-0015 | 3108364 | dependency_pending | full-data test baseline refresh using `RUN-0014` frozen thresholds | outputs/baselines/RUN-0015_hf5way_qwen_test_id_refresh_current |
| RUN-EXP-cascade_full | 3108360 | queued | full-data main learned run | outputs/experimental/RUN-EXP-cascade_full |
| RUN-EXP-flat_hidden_full | 3108362 | queued | full-data controlled hidden-state baseline | outputs/experimental/RUN-EXP-flat_hidden_full |
| RUN-EXP-dist_full | 3108363 | queued | full-data distribution-only comparator | outputs/experimental/RUN-EXP-dist_full |
| RUN-EXP-dist_full_v2 | 3108361 | queued | full-data weighted distribution comparator; highest wall-time risk | outputs/experimental/RUN-EXP-dist_full_v2 |

## Run Ledger
| run_id | date_utc | code_ref | config_path | dataset_manifest | split_scope | backbone_mode | baseline_name | acc | action_acc | macro_f1 | ece | brier | acceptance_pass | artifact_paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUN-0003 | 2026-02-28 19:27 UTC | working-tree | configs/class_medium_final.yaml | data/generated/pilots/pilot_3k_class_medium_real_vision.manifest.json | train+val+test_id+ood | n/a | materialize_vision_corrupt | N/A | N/A | N/A | N/A | N/A | pending | data/generated/pilots/pilot_3k_class_medium_real_vision.jsonl;data/generated/vision_corrupt/class_medium/pilot_3k |
| RUN-0002 | 2026-02-28 18:53 UTC | working-tree | configs/class_medium.yaml | data/generated/conflict_suite_class_medium.manifest.json | train+val+test_id+ood | n/a | dataset_build_class_medium | N/A | N/A | N/A | N/A | N/A | pending | data/interim/base_examples_class_medium.jsonl;data/generated/conflict_suite_class_medium.jsonl;data/generated/pilots/pilot_3k_class_medium.jsonl |
| RUN-0000 | 2026-02-24 18:45 UTC | working-tree | TBD | TBD | ID | mock | placeholder | TBD | TBD | TBD | TBD | TBD | no | TBD |
| RUN-0001 | 2026-02-24 19:31 UTC | working-tree | configs/cpu_local.yaml | data/generated/conflict_suite_full.manifest.json | ID+OOD | mock | two_pass_self_consistency | 0.0000 | 0.0000 | 0.7083 | 0.3164 | 0.3252 | no | artifacts/baselines/smoke |

## HF-First Quest Run Ledger
| run_id | date_utc | job_id | code_ref | config_path | dataset_manifest | split_scope | backbone_mode | baseline_suite | task_success | accuracy | coverage | accuracy_on_answered | acceptance_status | artifact_paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUN-0004 | 2026-03-06 08:23 UTC | 1305269 | working-tree | configs/hf_5way_qwen.yaml | data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json | test_id_smoke_8 | qwen2_5_vl_7b | backbone_direct+agreement_check+confidence_threshold+probe_heuristic | see report/main_table.md | see report/main_table.md | see report/main_table.md | see report/main_table.md | passed | outputs/baselines/RUN-0004_hf5way_qwen_test_id_preflight/summary.json;outputs/baselines/RUN-0004_hf5way_qwen_test_id_preflight/report/main_table.md |
| RUN-0005 | 2026-03-06 09:50 UTC | 1305301 | working-tree | configs/hf_5way_qwen.yaml | data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json | test_id | qwen2_5_vl_7b | backbone_direct+agreement_check+confidence_threshold+probe_heuristic | see report/main_table.md | see report/main_table.md | see report/main_table.md | see report/main_table.md | superseded | outputs/baselines/RUN-0005_hf5way_qwen_test_id/summary.json;outputs/baselines/RUN-0005_hf5way_qwen_test_id/report/main_table.md |
| RUN-0006 | 2026-03-07 09:32 UTC | 1571418 | working-tree | configs/hf_5way_qwen_runtime_normalized.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.manifest.json | val | qwen2_5_vl_7b | threshold_tuning_confidence+probe | see tuned_thresholds.json | see tuned_thresholds.json | see tuned_thresholds.json | see tuned_thresholds.json | completed_interim | outputs/baselines/RUN-0006_hf5way_qwen_val_tuning/tuned_thresholds.json;outputs/baselines/RUN-0006_hf5way_qwen_val_tuning/confidence_threshold_sweep.json;outputs/baselines/RUN-0006_hf5way_qwen_val_tuning/probe_heuristic_sweep.json |
| RUN-0007 | 2026-03-07 10:54 UTC | 1571420 | working-tree | configs/hf_5way_qwen_runtime_normalized.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.manifest.json | test_id | qwen2_5_vl_7b | backbone_direct+agreement_check+confidence_threshold+probe_heuristic | see report/main_table.md | see report/main_table.md | see report/main_table.md | see report/main_table.md | completed_interim | outputs/baselines/RUN-0007_hf5way_qwen_test_id_tuned/summary.json;outputs/baselines/RUN-0007_hf5way_qwen_test_id_tuned/report/main_table.md;outputs/baselines/RUN-0007_hf5way_qwen_test_id_tuned/report/per_category_task_success.md;outputs/baselines/RUN-0007_hf5way_qwen_test_id_tuned/report/c2_diagnostics.md |
| RUN-0009 | 2026-03-07 18:08 UTC | 1637025 | working-tree | configs/hf_5way_qwen_caption_derived.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_caption_derived_20260307.manifest.json | val | qwen2_5_vl_7b | threshold_tuning_confidence+probe | pending | pending | pending | pending | pending | outputs/baselines/RUN-0009_hf5way_qwen_val_tuning_caption_derived/tuned_thresholds.json;outputs/baselines/RUN-0009_hf5way_qwen_val_tuning_caption_derived/confidence_threshold_sweep.json;outputs/baselines/RUN-0009_hf5way_qwen_val_tuning_caption_derived/probe_heuristic_sweep.json |
| RUN-0010 | 2026-03-07 18:08 UTC | 1637026 | working-tree | configs/hf_5way_qwen_caption_derived.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_caption_derived_20260307.manifest.json | test_id | qwen2_5_vl_7b | backbone_direct+agreement_check+confidence_threshold+probe_heuristic | pending | pending | pending | pending | pending_dependency_on_RUN-0009 | outputs/baselines/RUN-0010_hf5way_qwen_test_id_tuned_caption_derived/summary.json;outputs/baselines/RUN-0010_hf5way_qwen_test_id_tuned_caption_derived/report/main_table.md;outputs/baselines/RUN-0010_hf5way_qwen_test_id_tuned_caption_derived/report/c2_diagnostics.md |

## CARM GPU Run Ledger
| run_id | date_utc | job_id | code_ref | config_path | dataset_manifest | split_scope | stage | task_success | action_accuracy | action_macro_f1 | status | artifact_paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUN-0008-train | 2026-03-07 10:58 UTC | 1571408 | working-tree | configs/hf_5way_qwen_runtime_normalized.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.manifest.json | train+val | train_carm | pending | pending | pending | out_of_memory | outputs/carm/RUN-0008_hf5way_qwen_runtime_normalized_train/slurm-1571408.err;outputs/carm/RUN-0008_hf5way_qwen_runtime_normalized_train/train_history.jsonl |
| RUN-0008-eval | 2026-03-07 10:58 UTC | 1571410 | working-tree | configs/hf_5way_qwen_runtime_normalized.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.manifest.json | test_id | evaluate_carm | pending | pending | pending | blocked_by_training_oom | outputs/carm/RUN-0008_hf5way_qwen_runtime_normalized_eval_test_id/slurm-1571410.out;outputs/carm/RUN-0008_hf5way_qwen_runtime_normalized_eval_test_id/slurm-1571410.err |
| RUN-0011-train | 2026-03-07 18:08 UTC | 1637027 | working-tree | configs/hf_5way_qwen_caption_derived_lowmem.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_caption_derived_20260307.manifest.json | train+val | train_carm | pending | pending | pending | pending | outputs/carm/RUN-0011_hf5way_qwen_caption_derived_train/slurm-1637027.out;outputs/carm/RUN-0011_hf5way_qwen_caption_derived_train/slurm-1637027.err |
| RUN-0011-eval | 2026-03-07 18:08 UTC | 1637028 | working-tree | configs/hf_5way_qwen_caption_derived_lowmem.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_caption_derived_20260307.manifest.json | test_id | evaluate_carm | pending | pending | pending | pending_dependency_on_RUN-0011-train | outputs/carm/RUN-0011_hf5way_qwen_caption_derived_eval_test_id/slurm-1637028.out;outputs/carm/RUN-0011_hf5way_qwen_caption_derived_eval_test_id/slurm-1637028.err |
| RUN-CTRL-0001 | 2026-03-11 03:46 UTC | 2391154 | working-tree | configs/hf_5way_qwen_caption_derived_10pct_protocol_family_lowmem.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | train+val+test_id | train_then_eval_test_id | 0.6098 | 0.5059 | 0.4633 | completed_success | outputs/carm/RUN-CTRL-0001_10pct_protocol/train/train_history.jsonl;outputs/carm/RUN-CTRL-0001_10pct_protocol/eval_test_id/metrics.json;outputs/carm/RUN-CTRL-0001_10pct_protocol/eval_test_id/per_example_predictions.jsonl |

## Experimental Small-Run Ledger
| run_id | date_utc | code_ref | config_path | dataset_manifest | split_scope | backbone_mode | stage | task_success_revised | action_accuracy | status | artifact_paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUN-EXP-cascade_10pct | 2026-03-16 18:55 UTC | working-tree | configs/cascade_carm_10pct.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | sampled train=3148,val=674,test_id=674 | qwen2_5_vl_7b | CascadeCARMHeads (139-dim: hidden_states+phi_v+phi_t+phi_cross, ep3 best val=0.731, early-stop p=3/3 at ep6) | 0.7344 | 0.6766 | completed_success | outputs/experimental/RUN-EXP-cascade_10pct/baseline_comparison.md;outputs/experimental/RUN-EXP-cascade_10pct/structured_carm/test/metrics.json |
| RUN-EXP-dist_10pct_v2 | 2026-03-16 19:30 UTC | working-tree | configs/distribution_carm_10pct_v2.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | sampled train=3148,val=674,test_id=674 | qwen2_5_vl_7b | DistributionCARMHeads + action_class_weights=[1,3,1,1], 15 epochs max (ep12 best val=0.648) | 0.6320 | 0.5638 | completed_success | outputs/experimental/RUN-EXP-dist_10pct_v2/baseline_comparison.md;outputs/experimental/RUN-EXP-dist_10pct_v2/structured_carm/test/metrics.json;outputs/experimental/RUN-EXP-dist_10pct_v2/ANALYSIS.md |
| RUN-EXP-dist_10pct | 2026-03-16 15:35 UTC | working-tree | configs/distribution_carm_10pct.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | sampled train=3148,val=674,test_id=674 | qwen2_5_vl_7b | distribution_cascade_10pct (DistributionCARMHeads, 110-dim input, 5 epochs early-stop) | 0.5994 | 0.5163 | completed_success | outputs/experimental/RUN-EXP-dist_10pct/baseline_comparison.md;outputs/experimental/RUN-EXP-dist_10pct/structured_carm/test/metrics.json;outputs/experimental/RUN-EXP-dist_10pct/ANALYSIS.md |
| RUN-EXP-0007 | 2026-03-11 03:46 UTC | working-tree | configs/experimental_10pct_qwen_protocol_family_rerun.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | sampled train=3148,val=674,test_id=674; protocol counts C1=900,C2=900,C3=900,C4=900,C5=896 | qwen2_5_vl_7b | revised_semantics_10pct_gpu_four_head | 0.5682 | 0.3961 | completed_success | outputs/experimental/RUN-EXP-0007_10pct_qwen_protocol/baseline_comparison.md;outputs/experimental/RUN-EXP-0007_10pct_qwen_protocol/structured_carm/test/metrics.json;outputs/experimental/RUN-EXP-0007_10pct_qwen_protocol/structured_carm/test/per_example_predictions.jsonl |
| RUN-EXP-0006 | 2026-03-11 03:40 UTC | working-tree | configs/experimental_preflight_qwen_protocol_family.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | sampled train=15,val=15,test_id=15 | qwen2_5_vl_7b | multimodal_gpu_preflight | see baseline_comparison.md | see structured_carm/test/metrics.json | preflight_completed_real_qwen_multimodal | outputs/experimental/RUN-EXP-0006_preflight_qwen_protocol/baseline_comparison.md;outputs/experimental/RUN-EXP-0006_preflight_qwen_protocol/structured_carm/test/metrics.json;outputs/experimental/RUN-EXP-0006_preflight_qwen_protocol/submission_metadata.txt |
| RUN-EXP-0005 | 2026-03-11 03:31 UTC | working-tree | configs/experimental_preflight_qwen_protocol_family.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | sampled train=15,val=15,test_id=15 | qwen2_5_vl_7b | multimodal_gpu_preflight | pending | pending | invalidated_failed_job_2390951_hf_rate_limited | outputs/experimental/RUN-EXP-0005_preflight_qwen_protocol/slurm-2390951.out;outputs/experimental/RUN-EXP-0005_preflight_qwen_protocol/submission_metadata.txt |
| RUN-EXP-0004 | 2026-03-11 03:14 UTC | working-tree | configs/experimental_10pct_qwen_protocol_family.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json | sampled train=3148,val=674,test_id=674; protocol counts C1=900,C2=900,C3=900,C4=900,C5=896 | qwen2_5_vl_7b | revised_semantics_10pct_gpu_four_head | pending | pending | invalidated_failed_job_2390788_misdiagnosed_loader_error | outputs/experimental/RUN-EXP-0004_10pct_qwen_protocol/failure_report.md;outputs/experimental/RUN-EXP-0004_10pct_qwen_protocol/baseline_comparison.md;outputs/experimental/RUN-EXP-0004_10pct_qwen_protocol/submission_metadata.txt |
| RUN-EXP-0003 | 2026-03-11 02:54 UTC | working-tree | configs/experimental_10pct_qwen.yaml | data/cache/hf_5way/prepared/carm_vqa_5way_10pct_seed7.manifest.json | sampled train=3148,val=674,test_id=674 | qwen2_5_vl_7b | revised_semantics_10pct_gpu | pending | pending | invalidated_canceled_job_2390150 | outputs/experimental/RUN-EXP-0003_10pct_qwen/data_sanity_report.md;outputs/experimental/RUN-EXP-0003_10pct_qwen/category_mapping_checks.json;outputs/experimental/RUN-EXP-0003_10pct_qwen/sbatch_command.txt;outputs/experimental/RUN-EXP-0003_10pct_qwen/submission_metadata.txt |
| RUN-EXP-0001 | 2026-03-10 06:02 UTC | working-tree | configs/experimental_small_debug.yaml | data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json | sampled train=60,val=30,test_id=30 | deterministic_debug_backbone | revised_semantics_preflight | 0.0833 | 0.2500 | preflight_completed_debug_only | outputs/experimental/RUN-EXP-0001_small_debug_preflight/failure_report.md;outputs/experimental/RUN-EXP-0001_small_debug_preflight/baseline_comparison.md;outputs/experimental/RUN-EXP-0001_small_debug_preflight/structured_carm/test/metrics.json |
| RUN-EXP-0002 | 2026-03-10 13:56 UTC | working-tree | configs/experimental_small_qwen.yaml | data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json | sampled train=15,val=15,test_id=15 | qwen2_5_vl_7b | revised_semantics_small_real_run | 0.3333 | 0.2500 | completed_real_small_run | outputs/experimental/RUN-EXP-0002_small_qwen/failure_report.md;outputs/experimental/RUN-EXP-0002_small_qwen/baseline_comparison.md;outputs/experimental/RUN-EXP-0002_small_qwen/structured_carm/test/metrics.json |

## Best-So-Far
| split_scope | baseline_name | run_id | acc | action_acc | macro_f1 | ece | brier | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID | placeholder | RUN-0000 | TBD | TBD | TBD | TBD | TBD | Bootstrap row until first real run. |
| ID+OOD | backbone_direct | RUN-0001 | 0.0476 | 0.1429 | 0.0000 | 0.0952 | 0.2500 | Smoke run on sample data fixture. |

## Acceptance Gate Checklist
| gate | metric | status | notes |
| --- | --- | --- | --- |
| Answer quality | Accuracy (overall, consistent/conflict, per family) | pending | Align with [PLAN.md](PLAN.md) Section 7.1. |
| Arbitration quality | Action accuracy | pending | Align with [PLAN.md](PLAN.md) Section 7.1. |
| Conflict recognition | Conflict-type macro F1 | pending | Align with [PLAN.md](PLAN.md) Section 7.1. |
| Selective prediction | Risk-coverage curves (ID and OOD) | pending | Align with [PLAN.md](PLAN.md) Section 7.1. |
| Reliability behavior | Monotonicity and calibration | pending | Align with [PLAN.md](PLAN.md) Section 7.1. |

## CASCADE CARM Results Summary (2026-03-16)

**RUN-EXP-cascade_10pct** achieved the best result in Phase A: TaskSuccess=**0.7344**, ActionAcc=0.6766.

Baseline comparison (test set, n=674):
| Predictor | TaskSuccess | Delta vs best-non-learned | Coverage | AccOnAnswered |
|-----------|-------------|---------------------------|---------|---------------|
| backbone_direct | 0.487 | — | 1.000 | 0.641 |
| agreement_check (best non-learned) | 0.577 | — | 0.430 | 0.752 |
| confidence_threshold | 0.494 | — | 0.991 | 0.645 |
| probe_heuristic | 0.460 | — | 1.000 | 0.596 |
| prompt_only_abstain | 0.439 | — | 0.487 | 0.701 |
| dist CARM v1 | 0.599 | +0.022 | — | — |
| **cascade CARM** | **0.734** | **+0.157 (+27.2%)** | 0.564 | **0.847** |

Per-category test breakdown (n=135 each except C5=134):
| Category | ActionAcc | TaskSuccess | Abstain% | Note |
|----------|-----------|------------|---------|------|
| C1 (consistent/require_agreement) | 0.667 | 0.807 | 11.9% | strong |
| C2 (trust_vision) | 0.741 | 0.807 | 11.9% | strong — fixed dist v1 collapse (was 0.237) |
| C3 (trust_text) | 0.526 | 0.511 | 40.0% | partial — dist v1 was 0.000, still weakest |
| C4 (contradictory/abstain) | 0.659 | 0.756 | 75.6% | good abstention |
| C5 (both_weak/abstain) | 0.791 | 0.791 | 79.1% | strong |

Key architectural insight: CascadeCARMHeads with frozen backbone hidden_states (128-dim) as input dramatically outperforms DistributionCARMHeads (which uses token distributions). The hidden states carry richer vision_info signal (val vision_info_acc 0.893 vs 0.651 for dist v1). The trust_text (C3) weakness at 51.1% is the main remaining gap.

RUN-EXP-dist_10pct_v2 (with action_class_weights=[1,3,1,1]) is still running — expected to add further C3 signal when complete.

## Known Regressions and Open Issues
- The current canonical prepared file has `31,463 / 6,743 / 6,742` examples by split, but the historical full-baseline artifact `RUN-0013_hf5way_qwen_test_id_tuned_caption_derived_refresh` logged `6747` test examples; that artifact is therefore provisional until refreshed.
- The archived `Flat Hidden` ablation used the 10% subset file directly and ended up with `3,146` rather than `3,148` train examples. It is useful evidence but not a controlled same-split comparison.
- `RUN-EXP-cascade_seed1`, `seed2`, and `seed3` were canceled by wall time, not invalidated by logic errors.
- `llava-hf/llava-v1.6-8b` remains out of scope for reportable evidence because `LlavaNextAdapter` is still a stub.
- The upstream `nbso/carm-vqa-5way` HF revision still lacks an explicit source-provided contradiction `text_supported_target` field, but the refreshed local caption-derived export now fills `8,931 / 8,992` C4 text targets (`99.32%`) and leaves the remaining `61` explicitly null with `text_supported_target_source=missing_after_caption_rule`.
- `RUN-0005` is superseded because it used the stale prepared export with the pre-HF local `C2/C4` numbering drift and emitted null contradiction diagnostic aggregates.
- `RUN-0006`/`RUN-0007`/`RUN-0008` use the local runtime-normalized fallback manifest, which corrects contradiction action labels without fabricating missing contradiction text targets. These runs remain interim for contradiction text diagnostics until HF is republished.
- `RUN-0009`/`RUN-0010`/`RUN-0011` are the first reruns against the caption-derived manifest `carm_vqa_5way_caption_derived_20260307.*`; until they complete, `RUN-0007` remains the latest finished baseline summary artifact.
- `RUN-0006` selected `confidence_threshold=0.9304117634483301` and `probe_both_uncertain_threshold=0.5384113192558289` on `val`.
- `RUN-0007` finished successfully, but `report/c2_diagnostics.md` remains `n/a` for all baselines because the fallback dataset still has no explicit contradiction text target.
- `RUN-0008-train` failed with Slurm `OUT_OF_MEMORY` after `02:38:10` on `qgpu2007`, so no `carm_heads.pt` checkpoint or eval metrics were produced.
- `RUN-0004` executed on an H100 80GB (`qgpu3020`) and passed the real-Qwen smoke gate before the full run was launched.
- `RUN-0005` executed on an A100 40GB (`qgpu0207`) and completed with full `6755`-row outputs for all four baselines.
- `RUN-EXP-0003` was canceled and invalidated after discovering that the first 10% subset was stratified on the older joint-info buckets and the running model still used the older joint-info-plus-reliability head.
- `RUN-EXP-0004` failed under a misleading one-line loader error and is superseded by the local-cache-first rerun chain `RUN-EXP-0005` -> `RUN-EXP-0006` -> `RUN-EXP-0007`.
- `RUN-EXP-0005` failed because unauthenticated HF Hub metadata requests hit HTTP `429` on the GPU node during multimodal preflight; this motivated the local-cache-first Qwen loader patch.
- `RUN-EXP-0006` completed the real multimodal GPU preflight successfully: both opt-in Qwen tests passed, all five baselines executed, and the structured four-head experimental path finished on the `15/15/15` subset.
- `RUN-EXP-0007` completed successfully on the corrected protocol-category-stratified 10% subset. The structured four-head model reached `task_success_revised=0.5682`, `action_accuracy=0.3961`, `relation_accuracy=0.4733`, `vision_info_accuracy=0.8501`, and `text_info_accuracy=0.5089`.
- `RUN-CTRL-0001` completed successfully on the same 10% subset. The old action-only control reached `task_success=0.6098`, `action_accuracy=0.5059`, and `action_macro_f1=0.4633`, outperforming the structured four-head model on final decision quality in this run.
- `RUN-EXP-cascade_10pct` (job 2946687, qgpu2010) completed successfully. CascadeCARMHeads with Qwen hidden_states input achieved test TaskSuccess=0.734, ActionAcc=0.677. Best val=0.731 at epoch 3; trained 6 epochs total (early-stop at patience=3). C2 TaskSuccess=0.807 (fixed), C3 TaskSuccess=0.511 (partial recovery from 0.000 in dist v1). Artifacts under outputs/experimental/RUN-EXP-cascade_10pct/.
- `RUN-EXP-dist_10pct_v2` (job 2946693, qgpu2004) completed 2026-03-16 19:30 UTC. Ran all 15 max_epochs; best val=0.648 at epoch 12. Test TaskSuccess=0.632 (+3.3pp over dist v1). C3 recovered partially (0.000→0.274) but still much weaker than cascade (0.511). C4/C5 strong (0.822/0.851). Confirms class weights alone insufficient vs hidden-state architecture. Artifacts: outputs/experimental/RUN-EXP-dist_10pct_v2/ANALYSIS.md.
- Non-ablation analysis artifacts for the completed 10% stage are frozen under `outputs/analysis/RUN-ANALYSIS-0001_10pct_protocol/`, including the figure bundle, failure examples, metrics snapshot, and `analysis_summary.md`.
- Ablations are intentionally deferred for this stage by user request.
- Legacy rows `RUN-0000` through `RUN-0003` remain historical context and are not the active HF-first Qwen baseline protocol.
- Governance bootstrap decision: [DEC-0001 Governance Artifacts Bootstrap](LOG.md#dec-0001-governance-artifacts-bootstrap).
- Phase A rebuild implementation logged at [DEC-0002 Phase A Rebuild Implementation](LOG.md#dec-0002-phase-a-rebuild-implementation).
- Class-medium profile and data build logged at [DEC-0003 Class-Medium Profile and Construction Throughput Patch](LOG.md#dec-0003-class-medium-profile-and-construction-throughput-patch).
- Vision materialization reproducibility policy logged at [DEC-0004 Reproducible Vision Materialization and Subset Image Acquisition](LOG.md#dec-0004-reproducible-vision-materialization-and-subset-image-acquisition).
- HF-first Quest submission path logged at [DEC-0005 HF-First Quest Qwen Baseline Submission Path](LOG.md#dec-0005-hf-first-quest-qwen-baseline-submission-path).
- Finalized C2 storage and runtime semantics logged at [DEC-0007 Finalized C2 Protocol Storage and Runtime Semantics](LOG.md#dec-0007-finalized-c2-protocol-storage-and-runtime-semantics).
- Frozen threshold-tuning workflow logged at [DEC-0008 Locked Validation Threshold Tuning and Frozen Threshold Artifacts](LOG.md#dec-0008-locked-validation-threshold-tuning-and-frozen-threshold-artifacts).
- Runtime-normalized fallback execution policy logged at [DEC-0009 Runtime-Normalized HF Fallback for Execution While Upstream C2 Targets Are Missing](LOG.md#dec-0009-runtime-normalized-hf-fallback-for-execution-while-upstream-c2-targets-are-missing).
