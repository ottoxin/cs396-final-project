# Proposal Technical Review and Spec Hardening

## High-priority fixes implemented

1. Reliability supervision is now concrete.
- `r_v, r_t` targets are derived from `evidence_modality`, `corrupted_modality`, and `severity`.
- Implementation: `/Users/hao/final-project/cs396-final-project/carm/data/labeling.py`.

2. Oracle action labeling is independent and deterministic.
- Action labels are generated from an explicit table over `(evidence_modality, corrupted_modality)`.
- Implementation: `/Users/hao/final-project/cs396-final-project/carm/data/labeling.py`.

3. `require_agreement` matching policy is specified.
- Structured answers use deterministic normalization equality.
- Free-form answers use token-overlap gate + semantic threshold.
- Implementation: `/Users/hao/final-project/cs396-final-project/carm/models/policy.py`.

## Medium-priority protocol fixes implemented

4. Conflict-suite shortcut reduction.
- Added adversarial filtering and similarity-balanced sampling to avoid trivial lexical cues.
- Implementation: `/Users/hao/final-project/cs396-final-project/carm/data/construction.py`.

5. Generalization leakage controls.
- Split checks enforce unique `example_id`, image-source disjointness, and template disjointness.
- Immutable manifest hash is emitted per split.
- Implementation: `/Users/hao/final-project/cs396-final-project/carm/data/integrity.py`.

6. Calibration acceptance criteria are explicit.
- Added thresholds for monotonicity violation rate, ECE, and Brier score.
- Implementation: `/Users/hao/final-project/cs396-final-project/carm/eval/evaluator.py`, `/Users/hao/final-project/cs396-final-project/configs/default.yaml`.
