# Auto Review Log — CARM Project

Initiated: 2026-03-16

## Round 1 (2026-03-16)

### Status: Implementation in Progress

The Codex MCP tool is not available in this environment. Proceeding with self-review and direct implementation based on deep code analysis.

### Self-Assessment (Senior ML Reviewer)
- Score: 4/10 current state
- Verdict: Not ready for submission yet — but a clear path exists to 7/10+

### Key Criticisms
1. **Structured model underperforms action-only control** (action_acc 0.396 vs 0.506) — this is the headline negative result. A multi-task model that's worse than the simpler baseline raises a fundamental question: is the structured formulation justified?
2. **Feature deficiency**: probe features are 3-dim (entropy, margin, variance). Missing the most informative signal: cross-modal answer agreement. This is the critical signal for distinguishing C1 (agree) from C4 (contradict, both informative).
3. **Flat multi-task architecture**: all 4 heads compete with equal loss weight from the same trunk. There's no information flow from informativeness predictions to action, despite the obvious semantic hierarchy: inform→ relation → action.
4. **Training budget**: 1 epoch with patience=1 is far too short for multi-task learning to converge.
5. **Dataset updated (2026-03-16)**: RUN-CTRL-0002 on fresh dataset shows val_coverage=0.038 (vs 0.516 old) — possible regression in data quality or label distribution change.

### Proposed Fixes (Implementing Now)
1. Add cross-modal agreement features (5-dim) to `carm/models/features.py`
2. Add `CascadeCARMHeads` with hierarchical prediction in `carm/experimental/model.py`
3. Update trainer to support cascade model with cross-modal features
4. Update `StructuredCARMPredictor` to handle cascade model
5. Update `run_experimental_small_data.py` to select model type from config
6. Create `configs/cascade_carm_10pct.yaml` with epochs=10, patience=3, λ_action=3.0

### Actions Taken
- [x] Read all relevant code files (model.py, training.py, features.py, evaluation.py, backbone.py, labels.py)
- [x] Read all config files and GPU submission scripts
- [x] Analyzed current failure modes in detail
- [x] Implemented `extract_cross_modal_features` in `carm/models/features.py` (5-dim phi_cross)
- [x] Implemented `CascadeCARMHeads` + `CascadeCARMConfig` in `carm/experimental/model.py`
- [x] Updated `ExperimentalCARMHeads.forward` to accept `phi_cross=None` (uniform interface)
- [x] Updated `ExperimentalTrainer._forward_example` to compute and pass phi_cross
- [x] Updated `StructuredCARMPredictor.predict` to compute and pass phi_cross
- [x] Updated `run_experimental_small_data.py` to select `model.type: cascade` from config
- [x] Created `configs/cascade_carm_10pct.yaml` with epochs=10, patience=3, lambda_action=3.0
- [x] Created `configs/cascade_carm_preflight.yaml` for 15-example validation
- [x] All import + forward + backward tests pass

### Results
- Previous best (old dataset): action_acc=0.506, task_success=0.610 (action-only control)
- Fresh repull (RUN-CTRL-0002, today, new dataset): val_task_success=0.424, val_coverage=0.038 — **dataset update changed distribution significantly**
- Cascade model: pending preflight + full GPU run
- Next: submit `cascade_carm_preflight.yaml` as preflight, then `cascade_carm_10pct.yaml` for full run

### Status: Round 1 implementation complete. Waiting for user to run GPU jobs.

---

## Architecture Notes

### CascadeCARMHeads design (implementing)
```
Input features: [pooled_hidden(128), phi_v(3), phi_t(3), phi_cross(5)] = 139-dim
Trunk: Linear(139→128) + GELU

Stage 1: informativeness
  vision_info_logits = vision_info_head(trunk_out)  # Linear(128→2)
  text_info_logits   = text_info_head(trunk_out)    # Linear(128→2)

Stage 2: pairwise relation (conditioned on info)
  relation_input = cat([trunk_out, softmax(vision_info_logits), softmax(text_info_logits)])  # 132-dim
  relation_logits = relation_head(Linear(132→64)+GELU → Linear(64→4))

Stage 3: action (conditioned on info + relation)
  action_input = cat([trunk_out, softmax(vision_info_logits), softmax(text_info_logits), softmax(relation_logits)])  # 136-dim
  action_logits = action_head(Linear(136→64)+GELU → Linear(64→4))
```

### Cross-modal features (phi_cross, 5-dim)
```
agreement_binary: 1.0 if v_answer_text == t_answer_text else 0.0
dist_overlap: sum(min(v_dist, t_dist))  # soft agreement
v_max_prob: max(v_dist)
t_max_prob: max(t_dist)
both_confident_disagree: v_max_prob * t_max_prob * (1 - agreement_binary)  # C4 signal
```

This directly encodes:
- C1 signature: agreement_binary=1, high v/t max_prob
- C4 signature: agreement_binary=0, high both_confident_disagree
- C2 signature: high v_max_prob, low t_max_prob
- C3 signature: low v_max_prob, high t_max_prob
- C5 signature: low v/t max_prob
