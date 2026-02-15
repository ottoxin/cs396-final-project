# Implementation Blueprint (Single-GPU Practical, CPU-Runnable Locally)

## Public interfaces

### Dataset schema (`jsonl`)
Required keys:
- `example_id`, `image_path`, `text_input`, `question`, `gold_answer`
- `split` in `{train,val,test}`
- `conflict_type` in `{none,object,attribute,relation,count}`
- `corrupted_modality` in `{none,vision,text}`
- `corruption_family`, `severity`
- `evidence_modality` in `{vision_required,text_required,both,either}`
- `oracle_action` in `{trust_vision,trust_text,require_agreement,abstain}`

### Model output contract (`CARMOutput`)
- `conflict_logits: float[5]`
- `reliability: {r_v, r_t}`
- `action_logits: float[4]`
- `action`, `final_answer`, `abstained`

## Module map

- Data and labels:
  - `/Users/hao/final-project/cs396-final-project/carm/data/schema.py`
  - `/Users/hao/final-project/cs396-final-project/carm/data/labeling.py`
  - `/Users/hao/final-project/cs396-final-project/carm/data/construction.py`
  - `/Users/hao/final-project/cs396-final-project/carm/data/integrity.py`
- Model and policy:
  - `/Users/hao/final-project/cs396-final-project/carm/models/backbone.py`
  - `/Users/hao/final-project/cs396-final-project/carm/models/carm_model.py`
  - `/Users/hao/final-project/cs396-final-project/carm/models/policy.py`
- Training:
  - `/Users/hao/final-project/cs396-final-project/carm/train/losses.py`
  - `/Users/hao/final-project/cs396-final-project/carm/train/trainer.py`
- Evaluation:
  - `/Users/hao/final-project/cs396-final-project/carm/eval/baselines.py`
  - `/Users/hao/final-project/cs396-final-project/carm/eval/metrics.py`
  - `/Users/hao/final-project/cs396-final-project/carm/eval/evaluator.py`

## Acceptance targets

- Conflict accuracy improves vs backbone baseline while consistent accuracy drop is bounded.
- Action accuracy and conflict macro-F1 exceed probe-only heuristic.
- Monotonicity violation rate <= 0.15.
- ECE <= 0.12 and Brier <= 0.25.
