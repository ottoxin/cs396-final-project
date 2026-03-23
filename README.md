# CARM: Conflict-Aware Reasoning Module

CARM studies multimodal conflict handling for frozen MLLMs. The project frames image-text disagreement as a structured arbitration problem over four actions:
`trust_vision`, `trust_text`, `require_agreement`, and `abstain`.

The current paper build uses the `nbso/carm-vqa-5way` prepared corpus with the canonical full split:
- `31,463` train
- `6,743` validation
- `6,742` test

## Result Snapshot

On the full `test` split with `Qwen/Qwen2.5-VL-7B-Instruct`:
- `agreement_check`: `0.489` task success
- `Dist CARM v1`: `0.588` task success
- `Flat Hidden`: `0.651` task success
- `Cascade CARM`: `0.662` task success

The stronger class-weighted `Dist CARM v2 (+wt)` full-data rerun timed out at the cluster's `48h` wall-time before final test evaluation.

## Setup

Create the project environment with Python `3.12`:

```bash
/gpfs/software/bowtie2/2.5.4/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision
./.venv/bin/python -m pip install --upgrade -e . pytest accelerate
```

Quest note:
- the default Quest `python3` is `3.6.8`
- this project requires `>=3.10`
- use the explicit `python3.12` path above when creating the venv

## Paper Build

The `writeup/` directory is self-contained for compilation. It includes the main LaTeX source, bibliography, ACL style files, local table/figure assets, and the compiled PDF.

Build the paper with:

```bash
bash writeup/build_paper.sh
```

The output PDF is:
- `writeup/carm_final.pdf`

## Repository Layout

- `carm/`: core library code for features, experimental heads, training, and evaluation
- `configs/`: runnable experiment configs
- `scripts/`: dataset prep, baseline, training, and utility entrypoints
- `tests/`: pytest coverage for key training and experimental paths
- `writeup/`: self-contained ACL paper sources and compiled PDF
- `reports/`: planning notes, experiment logs, and internal workflow documents

Large local artifacts are intentionally not committed:
- `data/cache/`
- `outputs/`
- cluster-generated logs and downloaded dataset material outside paper assets

## Config Defaults

`configs/default.yaml` now points to:
- HF repo: `nbso/carm-vqa-5way`
- cache root: `data/cache/hf_5way`
- default backbone: `qwen2_5_vl_7b`

## Output and Storage Paths

- `outputs/` is the run-output root (metrics JSON, per-example predictions, checkpoints, logs). It is empty until you run train/eval/baselines.
- Baseline output example: `outputs/baselines/hf_5way_qwen/`.
- Baseline run summary file: `summary.json`.
- Train/eval scripts write wherever you pass `--output_dir`.
- HF dataset/materialized images are written under `--cache-root` by `scripts/prepare_hf_5way_dataset.py`.
- Default cache/download location is `data/cache/hf_5way/`, including:
- `data/cache/hf_5way/prepared/carm_vqa_5way.jsonl`
- `data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json`
- `data/cache/hf_5way/images/`
- Hugging Face `datasets` may also keep its own cache under `~/.cache/huggingface/` unless you set `HF_HOME`/`HF_DATASETS_CACHE`.
- `data/cache/` is the active HF-first local materialization area and should stay gitignored.
- Archived stale or incompatible historical runs are moved under `outputs/archive/` so they are not accidentally resumed or compared against the active flattened schema.

## Optional Legacy Paths (Deprecated)

The old local-build pipeline from raw VQAv2/COCO artifacts is archived and no longer part of the active workflow.

For the canonical 5-way dataset workflow, use `scripts/prepare_hf_5way_dataset.py`.

Legacy v1 predictions can be migrated once to v2 rows:

```bash
./.venv/bin/python scripts/migrate_predictions_v1_to_v2.py \
  --input_jsonl <legacy_v1_predictions.jsonl> \
  --output_jsonl <migrated_v2_predictions.jsonl>
```

Legacy scripts no longer in active use have been moved to:
- `archive/legacy_scripts_<timestamp>/`
- `archive/raw_coco_vqav2_data_manipulation_20260304/`
- `archive/docs_governance_20260304/`
- `archive/configs_legacy_20260304/`

## Testing

Default test run (lightweight, no forced real-model inference):

```bash
./.venv/bin/python -m pytest -q
```

Opt-in real Qwen inference test:

```bash
RUN_QWEN_INFERENCE_TESTS=1 ./.venv/bin/python -m pytest tests/test_qwen_inference_optin.py
```

This test exercises real free-generation inference for existence, count, and color prompts and checks the canonical parsed answers. Real-model runs remain required for reported baseline results; the opt-in test is a pre-release validation gate.
