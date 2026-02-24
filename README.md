# CARM: Conflict-Aware Reasoning Module

This repository is in Phase A rebuild mode and follows [PLAN.md](PLAN.md).

## Governance Files

- [PLAN.md](PLAN.md): source of truth for protocol decisions.
- [REPORT.md](REPORT.md): canonical run ledger.
- [WRITEUP.md](WRITEUP.md): scientific narrative of implementation progress and results.
- `AGENT.md` and `LOG.md`: optional local notes, intentionally gitignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Phase A Runtime Workflow

### 1) Download official data artifacts

Default includes annotations, captions, and images.

```bash
python3 scripts/download_datasets.py --root data/raw
```

### 2) Build clean base dataset from VQAv2+COCO

```bash
python3 scripts/build_base_dataset.py \
  --config configs/cpu_local.yaml \
  --output_jsonl data/interim/base_examples.jsonl
```

### 3) Generate full canonical Conflict Suite v1

```bash
python3 scripts/generate_conflict_suite.py \
  --config configs/cpu_local.yaml \
  --input_jsonl data/interim/base_examples.jsonl \
  --output_jsonl data/generated/conflict_suite_full.jsonl \
  --manifest_json data/generated/conflict_suite_full.manifest.json
```

### 4) Sample deterministic pilot subset

```bash
python3 scripts/sample_pilot_subset.py \
  --config configs/cpu_local.yaml \
  --input_jsonl data/generated/conflict_suite_full.jsonl \
  --output_jsonl data/generated/pilots/pilot_3k_base.jsonl \
  --manifest_json data/generated/pilots/pilot_3k_base.manifest.json
```

### 5) Validate split/integrity contract

```bash
python3 scripts/check_data_integrity.py \
  --config configs/cpu_local.yaml \
  --input_jsonl data/generated/pilots/pilot_3k_base.jsonl
```

### 6) Run Phase A baselines

```bash
python3 scripts/run_baselines.py \
  --config configs/cpu_local.yaml \
  --input_jsonl data/generated/pilots/pilot_3k_base.jsonl \
  --output_dir artifacts/baselines/pilot_3k
```

## Output Layout

- `data/raw/...`: downloaded official artifacts.
- `data/interim/base_examples.jsonl`: filtered clean base examples.
- `data/generated/conflict_suite_full.jsonl`: canonical full suite.
- `data/generated/conflict_suite_full.manifest.json`: full-suite manifest and distributions.
- `data/generated/pilots/*.jsonl`: derived pilot datasets.
- `artifacts/baselines/<run_id>/`: per-baseline predictions and metrics.

## Baselines in Phase A

- `backbone_direct`
- `prompt_verification`
- `uncertainty_threshold_abstain`
- `two_pass_self_consistency`
- `probe_only_heuristic`

## Mock-First Note

Phase A is mock-backbone first for protocol validation and reproducibility checks.
Real runnable adapters for Qwen2.5-VL-7B and LLaVA-NeXT are represented as stubs and are planned for the next implementation wave.

## Documentation Discipline

1. After each run with artifacts, append one row in `REPORT.md` Run Ledger.
2. After each meaningful implementation milestone, update `WRITEUP.md` with a short scientific narrative of what changed and why.
3. If a change affects protocol/scope, update `PLAN.md` in the same commit and reference the corresponding run context from `REPORT.md`.
4. Keep `AGENT.md` and `LOG.md` as local working notes only (not committed).

## Docs Contract Validation

```bash
python3 scripts/validate_docs_contract.py
python3 scripts/validate_docs_contract.py --smoke
```

The validator always requires `REPORT.md`. `LOG.md` is validated only when the file is present locally.

## Common Failure Modes

- Missing official JSON paths in `configs/default.yaml` `data.paths`.
- Empty base dataset because consistency filters are too strict for local subset.
- No examples in a selected split/filter for baseline execution.
- OOD-family/OOD-severity misconfiguration causing integrity validation failures.

## Reproducibility Checklist

- Fix and log seed (`configs/*.yaml`).
- Save and track dataset manifests.
- Save resolved config with each run.
- Log every run in `REPORT.md` and summarize major milestones in `WRITEUP.md`.
