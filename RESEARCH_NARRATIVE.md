# CARM Research Narrative — Auto-Review Pipeline Entry Point

## Project Summary

**CARM** (Conflict-Aware Reasoning Module) is a lightweight, model-agnostic arbitration layer for multimodal LLMs (MLLMs). Given an image `I`, caption/text `T`, and question `q`, CARM decides how to handle modal conflict among four actions:
- `trust_vision`: answer from image only
- `trust_text`: answer from caption only
- `require_agreement`: answer only if both modalities agree
- `abstain`: refuse to answer (for contradictory or both-uninformative cases)

This is a NeurIPS-format paper for CS396 (Northwestern). Backbone: **Qwen2.5-VL-7B** (frozen). Dataset: **44,982 rows** from VQAv2/COCO structured into five protocol categories (C1–C5).

## Dataset / Protocol Categories

| Cat | Vision state | Text state | Oracle action | Semantics |
|-----|-------------|------------|---------------|-----------|
| C1 | informative | informative | `require_agreement` | consistent |
| C2 | informative | uninformative (irrelevant caption) | `trust_vision` | text asymmetry |
| C3 | uninformative (irrelevant image) | informative | `trust_text` | vision asymmetry |
| C4 | informative | informative but contradictory | `abstain` | contradiction |
| C5 | uninformative | uninformative | `abstain` | both weak |

Training split: 31,484 rows. Val: 6,743. Test (test_id): 6,755. 10% subset used for current experiments: ~4,496 rows.

## Current Architecture

The current implementation has two model variants:

### Variant A: Old Action-Only CARM (Control — RUN-CTRL-0001)
- Input: pooled anchor hidden states (dim=128) + vision probe features (3-dim) + text probe features (3-dim) = 134-dim
- Single trunk: Linear(134→128) + GELU
- One head: `joint_action` (4-way softmax)
- Loss: cross-entropy on oracle action only
- Result: `task_success=0.6098`, `action_accuracy=0.5059`, `action_macro_f1=0.4633` (1 epoch, patience=1)

### Variant B: Structured Four-Head CARM (Main — RUN-EXP-0007)
- Same input: pooled anchor hidden states + probe features
- One shared trunk: Linear(134→128) + GELU
- Four parallel heads (all from same trunk output):
  - `vision_info_head`: 2-way (informative/uninformative)
  - `text_info_head`: 2-way (informative/uninformative)
  - `relation_head`: 4-way (consistent/contradictory/asymmetric/both_weak)
  - `action_head`: 4-way
- Loss: masked multi-task sum with equal weights λ=1.0 for all 4 tasks
- Result: `task_success=0.5682`, `action_accuracy=0.3961` — **UNDERPERFORMS the simpler model**

## Current Results (10% subset, test_id split)

| Method | Coverage | AccAnswered | TaskSuccess | ActionAcc |
|--------|----------|-------------|-------------|-----------|
| backbone_direct | 1.00 | 0.653 | 0.467 | — |
| agreement_check | 0.445 | 0.780 | 0.571 | — |
| confidence_threshold | 0.990 | 0.657 | 0.472 | — |
| probe_heuristic | 1.00 | 0.620 | 0.461 | — |
| prompt_only_abstain | 0.493 | 0.669 | 0.409 | 0.339 |
| **structured_carm (4-head)** | **0.596** | **0.833** | **0.568** | **0.396** |
| **old_action_only (control)** | **0.516** | **0.816** | **0.610** | **0.506** |

### Structured model intermediate metrics:
- `vision_info_accuracy`: 0.850 (good)
- `text_info_accuracy`: 0.509 (poor — barely above chance for 2-class)
- `relation_accuracy`: 0.473 (poor — 4-class, chance=0.25 so some signal but weak)
- `joint_info_accuracy`: 0.441
- C4 multimodal abstain rate: only 0.341 (should be near 1.0)
- C4 contradiction_error_rate: 0.415

## Key Failure Analysis

1. **Structured model underperforms action-only control on action accuracy** (0.396 vs 0.506)
   - Multi-task loss is hurting action prediction despite more supervision
   - Equal loss weights: 4 tasks compete, and intermediate tasks are noisy

2. **text_info_accuracy is only 0.509** (close to chance for binary)
   - Text probe features (3-dim: entropy, margin, variance of sampling) are insufficient to distinguish informative vs uninformative text
   - The text channel features don't encode the semantic content of the caption relative to the question

3. **relation_accuracy = 0.473**
   - 4-class problem; random chance = 0.25 so there's some signal
   - But distinguishing `contradictory` vs `consistent` requires more than uncertainty features
   - Need agreement/disagreement signals between modalities

4. **C4 abstain rate = 0.341** (should be ~1.0)
   - The model is not learning to abstain on contradictory cases
   - Contradiction requires understanding that both modalities are informative but disagree
   - Current features don't explicitly encode cross-modal answer agreement

5. **Training is only 1 epoch with patience=1** (extremely limited)
   - Config: `epochs=1, patience=1` for low-memory setting
   - The model barely trains before early stopping

## Root Cause Hypotheses

### H1: Architecture — Flat multi-task with no hierarchy
The four heads all read from the same trunk independently. There is no information flow from informativeness predictions to relation prediction, or from relation to action. A **hierarchical/cascade architecture** would be more principled:
```
features → info_head (vision_info, text_info)
         → [info_preds + features] → relation_head
         → [relation_pred + info_preds + features] → action_head
```
This matches the natural semantic hierarchy: know if modalities are informative → understand their relationship → decide action.

### H2: Feature deficiency — probe features don't encode cross-modal agreement
Current 3-dim probe features (entropy, margin, variance) are purely unimodal uncertainty. Missing:
- **Cross-modal answer agreement**: do the vision-only and text-only answers match?
- **Answer identity features**: soft match between vision answer and text answer
- **Family-conditioned features**: question family (existence/count/color) affects what "agreement" means

### H3: Loss imbalance — intermediate tasks hurt action prediction
With equal λ=1.0 for all 4 tasks, the model spends 75% of its gradient budget on intermediate targets (vision_info, text_info, relation). These intermediate targets have noisy labels (especially C4 text_supported_target). Upweighting action loss (e.g., λ_action=3.0) while treating intermediate tasks as auxiliary regularizers would help.

### H4: Training budget — only 1 epoch is far too little
The low-memory config uses `epochs=1, patience=1` to avoid OOM. But 1 epoch on 3,148 training examples is insufficient for any multi-task model to converge. Need at least 5-10 epochs with patience=3.

## Proposed Improvements (Academically Valuable)

### Priority 1: Add cross-modal agreement features
Add explicit answer-agreement feature between vision-only and text-only probes:
- Binary: `v_answer == t_answer` (using family-normalized comparison)
- Soft: token overlap between vision answer and text answer
- Confidence-weighted: agreement signal weighted by min(v_conf, t_conf)

This directly provides the signal needed to distinguish C1 (agree) from C4 (disagree, both informative) from C2/C3 (asymmetric).

### Priority 2: Hierarchical/cascade architecture
Instead of flat parallel heads, cascade the predictions:
```
Stage 1: info_trunk → [vision_info_logits, text_info_logits]
Stage 2: relation_trunk([features; gumbel(info_logits)]) → relation_logits
Stage 3: action_trunk([features; gumbel(info_logits); gumbel(relation_logits)]) → action_logits
```
Use straight-through Gumbel-softmax to pass discrete signals through the cascade during training.

### Priority 3: Upweight action loss
Change loss weights: `λ_action=3.0`, `λ_relation=1.5`, `λ_vision_info=1.0`, `λ_text_info=1.0`.
The action prediction is the primary objective; intermediate tasks are regularizers.

### Priority 4: Increase training budget
Change config: `epochs=10`, `patience=3`. The model needs more training to benefit from multi-task supervision.

### Priority 5: Better feature representation
Expand probe features from 3-dim to 8-dim:
- entropy (1)
- top-1 margin (1)
- top-1 confidence (1)
- normalized rank of gold answer (1, if available)
- answer-agreement with other modality (1)
- confidence-weighted agreement (1)
- predicted-answer identity hash (coarse, 2-dim)

## Current Open Questions for This Stage

1. Can a hierarchical architecture (cascade) beat the flat multi-task model on action accuracy?
2. Do cross-modal agreement features provide sufficient signal to distinguish C1 vs C4?
3. What loss weighting gives the best trade-off between intermediate task accuracy and action prediction?
4. Can we get action_accuracy > 0.55 on the 10% subset?

## Academic Framing / Novelty Claims

The core novelty of CARM relative to prior work:
1. **Explicit arbitration layer** — prior work treats conflict as an end-task fine-tuning problem; CARM separates detection from arbitration
2. **Five-category protocol** with explicit informativeness/contradiction labels — more granular than prior binary conflict/no-conflict benchmarks (CLASH, etc.)
3. **Cascade architecture** (proposed) — hierarchical prediction from informativeness → relation → action, grounded in semantic dependency
4. **Calibrated abstention** — risk-coverage curves as first-class evaluation alongside accuracy

Relevant comparison points: CLASH [Popordanoska et al., 2025], uncertainty-based abstention [Wen et al., 2025], modality-bias decomposition [Zhang et al., 2025], LVLM hallucination surveys [Chen et al., 2025].

## Code Structure

```
cs396-final-project/
├── carm/
│   ├── experimental/
│   │   ├── model.py          # ExperimentalCARMHeads (current flat 4-head)
│   │   ├── training.py       # ExperimentalTrainer with masked multi-task loss
│   │   ├── evaluation.py     # Evaluation logic
│   │   ├── labels.py         # Label definitions (ACTION_LABELS, etc.)
│   │   └── baselines.py      # Baseline implementations
│   ├── models/
│   │   ├── backbone.py       # Qwen2.5-VL-7B adapter
│   │   └── features.py       # Probe feature extraction (currently 3-dim)
│   └── data/schema.py        # ConflictExample dataclass
├── configs/
│   └── experimental_10pct_qwen_protocol_family_rerun.yaml  # Current config
├── scripts/
│   ├── submit_experimental_quest.sh  # GPU submission
│   └── submit_carm_quest.sh
├── PLAN.md                    # Research plan
├── REPORT.md                  # Run ledger
├── NEW_PLAN.MD                # Detailed experimental plan
└── writeup/carm_proposal.tex  # LaTeX paper (NeurIPS format)
```

## Next Steps (for auto pipeline to implement)

In priority order:

1. **Add cross-modal answer-agreement features** to `carm/models/features.py` and update `ExperimentalCARMHeads` input size
2. **Implement hierarchical/cascade architecture** in `carm/experimental/model.py` as a new `CascadeCARMHeads` class
3. **Update loss weights** in training config and `ExperimentalTrainerConfig`
4. **Create new config** `configs/cascade_carm_10pct.yaml` with `epochs=10, patience=3, lambda_action=3.0`
5. **Run preflight** on 15-example subset to validate pipeline
6. **Submit GPU job** for full 10% training
7. **Update paper** (`writeup/carm_proposal.tex`) with new architecture and results

## Success Criteria

- `action_accuracy > 0.55` on test_id (beating old control at 0.5059)
- `task_success > 0.62` on test_id (beating current best of 0.6098)
- `relation_accuracy > 0.60` (coherent intermediate structure)
- C4 multimodal abstain rate > 0.70 (correct contradiction handling)

## GPU Setup (Quest HPC at Northwestern)

Jobs submitted via `sbatch`. Relevant scripts in `scripts/`. The project uses Qwen2.5-VL-7B loaded via HuggingFace transformers, inference-only backbone (frozen), CARM heads trained on GPU. Memory constraint: ~40-80GB GPU RAM.

```bash
# Submit training job
bash scripts/submit_experimental_quest.sh <config_path> <output_dir>
# or
sbatch scripts/submit_carm_quest.sh
```
