# CARM: Conflict-Aware Reasoning Module

This repository is in Phase A rebuild mode and follows [PLAN.md](PLAN.md).

## Governance Files

- [PLAN.md](PLAN.md): source of truth for protocol decisions.
- [WRITEUP.md](WRITEUP.md): scientific narrative of implementation progress and results.
- `REPORT.md`, `AGENT.md`, and `LOG.md`: optional local notes, intentionally gitignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Required only if you want to pull the prebuilt HF release:
pip install huggingface_hub
```

## Repository Layout

- `carm/data/`: dataset schema, VQAv2+COCO ingestion, conflict construction, integrity checks, sampling.
- `carm/eval/`: baselines, evaluator, and metric computations.
- `carm/models/`: mock backbone, adapter interfaces, registry, and CARM model modules.
- `carm/train/`: training dataset/loss/trainer scaffolding.
- `configs/`: protocol defaults and runtime profiles (`default`, `cpu_local`, `cloud_gpu`, `class_medium`, `class_medium_final`, ablations).
- `scripts/`: runnable CLIs for download/build/generate/sample/check/evaluate/validate-docs.
- `tests/`: unit and smoke tests for data pipeline, integrity, and evaluation contracts.
- `data/`: local datasets and generated artifacts (`raw/`, `interim/`, `generated/`, `clean/`).
- `artifacts/`: baseline outputs and run artifacts.
- `PLAN.md`: protocol source of truth.
- `REPORT.md`: local run ledger (gitignored in this repo).

## Fast Path: Pull Prebuilt Class-Medium Data

If you want to skip rebuild from raw VQAv2/COCO, pull the released artifact snapshot and install it directly into this repo's `data/generated/...` layout:

```bash
python3 scripts/download_datasets.py \
  --source release \
  --hf-revision main
```

Default release repo: `haohxin/cs396-final` (override with `--hf-repo-id` if needed).

For full baseline inference across all 21,000 pilot rows, also fetch official COCO train/val images in the same command:

```bash
python3 scripts/download_datasets.py \
  --source release \
  --hf-revision main \
  --with-official-images \
  --root data/raw
```

What this installs:
- `data/generated/pilots/pilot_3k_class_medium_real_vision.jsonl`
- `data/generated/pilots/pilot_3k_class_medium_real_vision.manifest.json`
- `data/generated/vision_corrupt/class_medium/pilot_3k/` (unless `--skip-release-images`)
- supporting manifests under `data/generated/` and `data/interim/`

Important source-data note:
- HF release snapshot does **not** redistribute original VQAv2/COCO annotation archives.
- Pilot `clean/swap_easy/swap_hard/text_edit` rows reference original COCO images under `data/raw/coco/...`.
- Use `--with-official-images` (above) or run `--source official` to download originals locally.

Dataset card (tracked in this repo):
- `DATASET_CARD.md`

Release snapshot also includes a copy at:
- `data/hf_release/cs396-final-dataset/README.md` (local cache after `--source release`)

## Phase A Runtime Workflow (Rebuild From Raw Data)

### 1) Download official VQAv2/COCO artifacts

Default includes annotations, captions, and images.

```bash
python3 scripts/download_datasets.py \
  --source official \
  --root data/raw
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

## Class-Medium Workflow (CS396 Scale)

This profile keeps the same protocol but caps base construction to `max_per_family=5000` for faster class-project iteration.

```bash
python3 scripts/build_base_dataset.py \
  --config configs/class_medium.yaml

python3 scripts/generate_conflict_suite.py \
  --config configs/class_medium.yaml \
  --input_jsonl data/interim/base_examples_class_medium.jsonl \
  --output_jsonl data/generated/conflict_suite_class_medium.jsonl \
  --manifest_json data/generated/conflict_suite_class_medium.manifest.json

python3 scripts/sample_pilot_subset.py \
  --config configs/class_medium.yaml \
  --input_jsonl data/generated/conflict_suite_class_medium.jsonl \
  --output_jsonl data/generated/pilots/pilot_3k_class_medium.jsonl \
  --manifest_json data/generated/pilots/pilot_3k_class_medium.manifest.json

python3 scripts/check_data_integrity.py \
  --config configs/class_medium.yaml \
  --input_jsonl data/generated/pilots/pilot_3k_class_medium.jsonl
```

### Optional: Materialize Real Pixel-Level Vision Corruption on Pilot

This step rewrites pilot vision-corrupt rows to point at deterministic occluded images and emits a reproducibility manifest with input/output hashes.

```bash
python3 scripts/materialize_vision_corrupt.py \
  --input_jsonl data/generated/pilots/pilot_3k_class_medium.jsonl \
  --output_jsonl data/generated/pilots/pilot_3k_class_medium_real_vision.jsonl \
  --output_image_dir data/generated/vision_corrupt/class_medium/pilot_3k \
  --manifest_json data/generated/pilots/pilot_3k_class_medium_real_vision.manifest.json \
  --jpeg_quality 90 \
  --download_missing_coco \
  --fingerprint_images
```

Current materialized pilot artifact snapshot (2026-02-28):
- `data/generated/pilots/pilot_3k_class_medium_real_vision.jsonl`: `21,000`
- `data/generated/vision_corrupt/class_medium/pilot_3k/`: `9,000` images
- `data/generated/pilots/pilot_3k_class_medium_real_vision.manifest.json`: deterministic hashes + image-dir fingerprint

Current class-medium artifact counts (2026-02-28):
- `data/interim/base_examples_class_medium.jsonl`: `15,000`
- `data/generated/conflict_suite_class_medium.jsonl`: `105,000`
- `data/generated/pilots/pilot_3k_class_medium.jsonl`: `21,000`

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
4. Keep `REPORT.md`, `AGENT.md`, and `LOG.md` as local working notes only (not committed).

## Docs Contract Validation

```bash
python3 scripts/validate_docs_contract.py
python3 scripts/validate_docs_contract.py --smoke
```

The validator supports local-only docs. It validates `REPORT.md` and/or `LOG.md` when present.

## Common Failure Modes

- Missing official JSON paths in `configs/default.yaml` `data.paths`.
- Empty base dataset because consistency filters are too strict for local subset.
- Running with `default.yaml` paths after class-medium generation and reading stale smoke-size files from `data/generated/conflict_suite_full.jsonl`.
- No examples in a selected split/filter for baseline execution.
- OOD-family/OOD-severity misconfiguration causing integrity validation failures.

## Recent Engineering Patch

`carm/data/construction.py` hard-swap donor search was optimized by:
- pre-indexing donors by `(family, answer_type)`
- caching noun-like tokenization used in Jaccard checks

This preserves generation behavior and significantly reduces runtime for medium/full suite construction.

## Reproducibility Checklist

- Fix and log seed (`configs/*.yaml`).
- Save and track dataset manifests.
- Save resolved config with each run.
- Log every run in `REPORT.md` and summarize major milestones in `WRITEUP.md`.

## Optional HF Release

For dataset release, publish these together to keep it reproducible:
- input manifest and generation manifests
- final JSONL (`pilot_3k_class_medium_real_vision.jsonl`)
- materialized image directory (`data/generated/vision_corrupt/class_medium/pilot_3k`)
- materialization manifest (`pilot_3k_class_medium_real_vision.manifest.json`)

Current dataset repo target: `haohxin/cs396-final`.

Example upload command:

```bash
hf upload haohxin/cs396-final . --repo-type=dataset
```
