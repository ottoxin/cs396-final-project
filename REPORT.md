# REPORT.md - Phase A Results Ledger

Purpose: canonical experiment results ledger for Phase A. This file is the truth source for run-level outputs and acceptance-gate status.

Last updated: 2026-02-24 19:31 UTC

## Phase Status Snapshot
- current_phase: Phase A
- latest_dataset_manifest_id: data/generated/conflict_suite_full.manifest.json
- latest_baseline_summary_artifact: artifacts/baselines/smoke/summary.json

## Run Ledger
| run_id | date_utc | code_ref | config_path | dataset_manifest | split_scope | backbone_mode | baseline_name | acc | action_acc | macro_f1 | ece | brier | acceptance_pass | artifact_paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUN-0000 | 2026-02-24 18:45 UTC | working-tree | TBD | TBD | ID | mock | placeholder | TBD | TBD | TBD | TBD | TBD | no | TBD |
| RUN-0001 | 2026-02-24 19:31 UTC | working-tree | configs/cpu_local.yaml | data/generated/conflict_suite_full.manifest.json | ID+OOD | mock | two_pass_self_consistency | 0.0000 | 0.0000 | 0.7083 | 0.3164 | 0.3252 | no | artifacts/baselines/smoke |

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

## Known Regressions and Open Issues
- No regressions recorded yet.
- Governance bootstrap decision: [DEC-0001 Governance Artifacts Bootstrap](LOG.md#dec-0001-governance-artifacts-bootstrap).
- Phase A rebuild implementation logged at [DEC-0002 Phase A Rebuild Implementation](LOG.md#dec-0002-phase-a-rebuild-implementation).
