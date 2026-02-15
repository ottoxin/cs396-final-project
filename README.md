# CARM: Conflict-Aware Reasoning Module

This repository implements a CPU-runnable, single-GPU-ready scaffold for CARM with:

- Typed dataset schema and deterministic split integrity checks
- Explicit oracle action labeling (`evidence_modality x corrupted_modality`)
- Reliability target construction for `r_v` and `r_t`
- CARM heads (conflict, reliability, action) over anchor states + probe features
- Action-conditioned generation with auditable suppression paths
- Baselines, evaluation metrics, and per-example prediction logging
- Unit + integration tests using `unittest`

## Quickstart (CPU local)

```bash
cd /Users/hao/final-project/cs396-final-project
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Pipeline commands

1. Generate conflict suite:
```bash
python3 scripts/generate_conflict_suite.py \
  --input_jsonl data/clean/examples.jsonl \
  --output_jsonl data/generated/conflict_suite.jsonl
```

2. Validate data integrity:
```bash
python3 scripts/check_data_integrity.py --input_jsonl data/generated/conflict_suite.jsonl
```

3. Train CARM heads:
```bash
python3 scripts/train_carm.py \
  --config configs/cpu_local.yaml \
  --train_jsonl data/generated/conflict_suite.jsonl \
  --output_dir artifacts/train_run
```

4. Evaluate CARM:
```bash
python3 scripts/evaluate_carm.py \
  --config configs/cpu_local.yaml \
  --input_jsonl data/generated/conflict_suite.jsonl \
  --output_dir artifacts/eval_run
```

5. Run baselines:
```bash
python3 scripts/run_baselines.py \
  --config configs/cpu_local.yaml \
  --input_jsonl data/generated/conflict_suite.jsonl \
  --output_dir artifacts/baselines
```

## Cloud-ready GPU profile

Use `configs/cloud_gpu.yaml` when moving to cloud compute. The code keeps backbone frozen in v1.
