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
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install datasets huggingface_hub
```

## HF-First Workflow

### 1) Prepare baseline-ready local outputs from HF dataset

```bash
python3 scripts/prepare_hf_5way_dataset.py \
  --hf-repo-id nbso/carm-vqa-5way \
  --hf-revision main \
  --cache-root data/cache/hf_5way
```

This writes:
- `data/cache/hf_5way/prepared/carm_vqa_5way.jsonl`
- `data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json`
- `data/cache/hf_5way/images/*.jpg`

### 2) Run baselines (Qwen)

```bash
python3 scripts/run_baselines.py \
  --config configs/hf_5way_qwen.yaml \
  --input_jsonl data/cache/hf_5way/prepared/carm_vqa_5way.jsonl \
  --output_dir outputs/baselines/hf_5way_qwen \
  --split all \
  --resume \
  --progress-every 500
```

## Repository Layout

- `carm/data/`: schema, transforms, integrity, and HF 5-way helpers.
- `carm/eval/`: baselines, evaluator, and metric computation.
- `carm/models/`: runnable Qwen adapter and model registry.
- `carm/train/`: CARM head training utilities.
- `scripts/`: dataset prep, baseline runs, train/eval CLIs.
- `configs/`: runtime profiles (`default`, `hf_5way_qwen`, legacy configs).
- `tests/`: unit and integration tests.

Tree view:

```text
cs396-final-project/
├── carm/
│   ├── data/
│   ├── eval/
│   ├── models/
│   ├── train/
│   └── utils/
├── configs/
│   ├── default.yaml
│   ├── hf_5way_qwen.yaml
│   └── ...
├── scripts/
│   ├── prepare_hf_5way_dataset.py
│   ├── run_baselines.py
│   ├── train_carm.py
│   ├── evaluate_carm.py
│   └── ...
├── tests/
├── data/
│   ├── clean/
│   └── cache/    # HF-first local materialization (gitignored)
├── outputs/      # run outputs (gitignored)
└── archive/
    └── local_data_<timestamp>/    # local-only archived raw/interim/generated/hf_release data
```

## Config Defaults

`configs/default.yaml` now points to:
- HF repo: `nbso/carm-vqa-5way`
- cache root: `data/cache/hf_5way`
- default backbone: `qwen2_5_vl_7b`

## Output and Storage Paths

- `outputs/` is the run-output root (metrics JSON, per-example predictions, checkpoints, logs). It is empty until you run train/eval/baselines.
- Baseline output example: `outputs/baselines/hf_5way_qwen/`.
- Train/eval scripts write wherever you pass `--output_dir`.
- HF dataset/materialized images are written under `--cache-root` by `scripts/prepare_hf_5way_dataset.py`.
- Default cache/download location is `data/cache/hf_5way/`, including:
- `data/cache/hf_5way/prepared/carm_vqa_5way.jsonl`
- `data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json`
- `data/cache/hf_5way/images/`
- Hugging Face `datasets` may also keep its own cache under `~/.cache/huggingface/` unless you set `HF_HOME`/`HF_DATASETS_CACHE`.
- `data/cache/` is the active HF-first local materialization area and should stay gitignored.

## Optional Legacy Paths (Deprecated)

The old local-build pipeline from raw VQAv2/COCO artifacts is archived and no longer part of the active workflow.

For the canonical 5-way dataset workflow, use `scripts/prepare_hf_5way_dataset.py`.

Legacy scripts no longer in active use have been moved to:
- `archive/legacy_scripts_<timestamp>/`
- `archive/raw_coco_vqav2_data_manipulation_20260304/`
- `archive/docs_governance_20260304/`

## Testing

Default test run (lightweight, no forced real-model inference):

```bash
pytest -q
```

Opt-in real Qwen inference test:

```bash
RUN_QWEN_INFERENCE_TESTS=1 pytest tests/test_qwen_inference_optin.py
```

Real-model runs remain required for reported baseline results; the opt-in test is a pre-release validation gate.
