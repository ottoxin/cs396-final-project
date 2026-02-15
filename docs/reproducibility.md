# Reproducibility

## Determinism
- Seed fixed via config (`seed`) and `set_global_seed`.
- Split manifests are hash-based for immutable split tracking.

## End-to-end commands

```bash
python3 scripts/generate_conflict_suite.py --input_jsonl data/clean/sample_clean.jsonl --output_jsonl data/generated/conflict_suite.jsonl
python3 scripts/check_data_integrity.py --input_jsonl data/generated/conflict_suite.jsonl
python3 scripts/train_carm.py --config configs/cpu_local.yaml --train_jsonl data/generated/conflict_suite.jsonl --output_dir artifacts/train_run
python3 scripts/evaluate_carm.py --config configs/cpu_local.yaml --input_jsonl data/generated/conflict_suite.jsonl --output_dir artifacts/eval_run --model_ckpt artifacts/train_run/carm_heads.pt --split test
python3 scripts/run_baselines.py --config configs/cpu_local.yaml --input_jsonl data/generated/conflict_suite.jsonl --output_dir artifacts/baselines
```
