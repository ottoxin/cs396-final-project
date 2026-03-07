# CARM: Conflict-Aware Reasoning Module

This repository now uses a Hugging Face first workflow.

Canonical dataset source:
- `nbso/carm-vqa-5way`
- current realized release size: `44,982` rows (single HF split before local deterministic split export)

Important count clarification:
- `150,582 / 15,207 / 18,801` in `WRITEUP.md` are upstream retained clean-base counts from official VQAv2+COCO filtering.
- `44,982` is the currently published HF refined-run corpus used for current baseline workflows.

## Setup

```bash
/gpfs/software/bowtie2/2.5.4/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision
./.venv/bin/python -m pip install --upgrade -e . pytest accelerate
```

Quest GPU baseline bootstrap on the login node:

```bash
/gpfs/software/bowtie2/2.5.4/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision
./.venv/bin/python -m pip install --upgrade -e . pytest accelerate
```

Why this interpreter path is explicit:
- the default Quest `python3` on login and compute nodes is `3.6.8`
- the project requires `>=3.10`
- a venv built against `/usr/bin/python3.12` on a login shell will not run on compute nodes here

## HF-First Workflow

### 1) Prepare baseline-ready local outputs from HF dataset

```bash
./.venv/bin/python scripts/prepare_hf_5way_dataset.py \
  --hf-repo-id nbso/carm-vqa-5way \
  --hf-revision main \
  --cache-root data/cache/hf_5way
```

This writes:
- `data/cache/hf_5way/prepared/carm_vqa_5way.jsonl`
- `data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json`
- `data/cache/hf_5way/images/*.jpg`

Protocol note:
- HF prep now requires an explicit C2 `text_supported_target` source field.
- If the active HF revision omits that field, prep fails loudly and writes a manifest with `status=failed`.
- For the current interim execution fallback in this repo, use `configs/hf_5way_qwen_runtime_normalized.yaml` plus `data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.jsonl`. That local artifact fixes stale C2 action labels but does not fabricate missing C2 text targets, so C2 text diagnostics remain unavailable and the run should not be treated as the final protocol-complete refresh.

### 2) Tune thresholded baselines on `val`

```bash
./.venv/bin/python scripts/tune_baseline_thresholds.py \
  --config configs/hf_5way_qwen_runtime_normalized.yaml \
  --input_jsonl data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.jsonl \
  --output_dir outputs/baselines/RUN-0006_hf5way_qwen_val_tuning \
  --split val
```

This writes:
- `outputs/baselines/RUN-0006_hf5way_qwen_val_tuning/tuned_thresholds.json`
- `outputs/baselines/RUN-0006_hf5way_qwen_val_tuning/confidence_threshold_sweep.json`
- `outputs/baselines/RUN-0006_hf5way_qwen_val_tuning/probe_heuristic_sweep.json`

### 3) Run locked baselines on `test_id`

```bash
./.venv/bin/python scripts/run_baselines.py \
  --config configs/hf_5way_qwen_runtime_normalized.yaml \
  --input_jsonl data/cache/hf_5way/prepared/carm_vqa_5way_runtime_normalized_20260307.jsonl \
  --output_dir outputs/baselines/RUN-0007_hf5way_qwen_test_id_tuned \
  --tuned-thresholds-json outputs/baselines/RUN-0006_hf5way_qwen_val_tuning/tuned_thresholds.json \
  --resume \
  --split test_id \
  --progress-every 500
```

These commands assume the project virtualenv. Do not rely on system `python3` on Quest.

Active baseline set in the runner:
- `backbone_direct`
- `agreement_check`
- `confidence_threshold`
- `probe_heuristic`

Each baseline uses the same flat evaluator contract:
- per-example core fields: input/gold metadata, `final_answer`, `abstained`, `confidence`, `correct`, `task_success`
- optional extra fields are appended for richer predictors and C2 diagnostics such as `pred_action`, `r_v`, `r_t`, `audit`, `c2_vision_only_correct`, `c2_text_only_correct`, and `c2_multimodal_abstained`

Main artifacts:
- per baseline directory:
  - `per_example_predictions.jsonl`
  - `metrics.json`
- run root:
  - `summary.json`
  - `applied_tuned_thresholds.json`
  - `report/main_table.csv`
  - `report/main_table.md`
  - `report/per_category_task_success.csv`
  - `report/per_category_task_success.md`
  - `report/c2_diagnostics.csv`
  - `report/c2_diagnostics.md`
  - `report/risk_coverage_task_success_curves.json`

The backbone now answers via free generation with family-specific prompting/parsing:
- existence: `Answer yes or no only.`
- count: `Answer with a single integer only.`
- color: `Answer with a single color word only.`

### 4) Rerender baseline report tables (optional)

```bash
./.venv/bin/python scripts/summarize_baselines_report.py \
  --baselines-root outputs/baselines/RUN-0007_hf5way_qwen_test_id_tuned \
  --target-coverage 0.8
```

`scripts/run_baselines.py` now writes the report automatically; rerender manually only if you need to rebuild the tables from existing per-example outputs.

## Repository Layout

- `carm/`: package code used by all runtime CLIs (data, models, train, eval, utils).
- `scripts/`: command-line entrypoints (prepare dataset, baselines, train, evaluate).
- `configs/`: active runtime configs (`default.yaml`, `hf_5way_qwen.yaml`).
- `tests/`: automated validation (`pytest`) for schema, mapping, policy, and integration.
- `data/`: local sample + cache root for prepared HF artifacts.
- `outputs/`: run outputs (metrics, per-example predictions, checkpoints/logs).
- `archive/`: local-only archived legacy scripts/configs/data.

Tree view:

```text
cs396-final-project/
├── carm/         # core library package
│   ├── data/     # schema + prep helpers
│   ├── eval/     # baselines + metrics + evaluator
│   ├── models/   # backbone adapters + CARM heads + policy
│   ├── train/    # dataset/loss/trainer
│   └── utils/    # config + seed helpers
├── configs/      # active runtime configs
│   ├── default.yaml
│   └── hf_5way_qwen.yaml
├── scripts/      # runnable CLIs
│   ├── prepare_hf_5way_dataset.py
│   ├── run_baselines.py
│   ├── summarize_baselines_report.py
│   ├── train_carm.py
│   └── evaluate_carm.py
├── tests/        # pytest suite (kept separate from runtime scripts)
├── data/
│   ├── sample/   # tiny local sample data
│   └── cache/    # HF-first local materialization (gitignored)
├── outputs/      # run outputs (gitignored)
└── archive/      # local-only legacy materials (gitignored)
```

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
