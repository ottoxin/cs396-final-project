# CARM Project Brief

**Conflict-Aware Reasoning Module for Multimodal LLMs**
CS396 Final Project — Northwestern University
Otto Xin & Nicholas Ornstein

---

## 1. Problem

Multimodal LLMs (MLLMs) fail silently when image and text inputs conflict. Given an image of a red ball and a caption saying "blue ball," leading models (LLaVA, InstructBLIP, Qwen-VL) respond confidently but inconsistently — sometimes trusting the image, sometimes the text, with no principled arbitration. This is not a perception failure; it is a missing arbitration mechanism.

We address this with **CARM**: a lightweight, model-agnostic arbitration layer that makes conflict handling explicit, testable, and interpretable.

---

## 2. Dataset: Five-Category Conflict Suite

Constructed from VQAv2 questions + COCO images/captions. **44,982 examples** across three question families (existence, count, attribute\_color) and five protocol categories:

| Category | Image | Text | Oracle action | Semantics |
|---|---|---|---|---|
| C1 | clean | clean | `require_agreement` | Consistent — answer only if both agree |
| C2 | clean | irrelevant | `trust_vision` | Text carries no signal |
| C3 | irrelevant | clean | `trust_text` | Image carries no signal |
| C4 | clean | contradictory | `abstain` | Both informative but disagree |
| C5 | irrelevant | irrelevant | `abstain` | Neither modality is informative |

**Split**: 31,484 train / 6,743 val / 6,755 test. Active experiments use a 10% stratified subset (~4,496 examples).

The five-category protocol is a core novelty: prior work (CLASH, etc.) uses binary conflict/no-conflict. Explicitly modeling *why* to abstain (C4: contradiction vs. C5: both weak) is more principled and enables richer evaluation.

---

## 3. Previous State of the Repo

### Architecture (Phase A)

Two model variants were trained and evaluated on the 10% subset:

**Variant A — Action-Only Control** (`model.type: flat`, `RUN-CTRL-0001`)
- Input: `[pooled_hidden_states(128), phi_v(3), phi_t(3)]` = 134-dim
- One shared trunk: `Linear(134→128) + GELU`
- One head: joint action (4-way)
- Loss: cross-entropy on oracle action only
- Training: 1 epoch, patience=1

**Variant B — Structured Four-Head** (`model.type: flat`, `RUN-EXP-0007`)
- Same 134-dim input
- Same shared trunk
- Four parallel heads: `vision_info`, `text_info`, `pairwise_relation`, `action`
- Loss: equal-weight multi-task (`λ=1.0` for all four tasks)
- Training: 1 epoch, patience=1

### Phase A Results (old dataset, 10% subset)

| Method | Coverage | Acc@Answered | TaskSuccess | ActionAcc |
|---|---|---|---|---|
| backbone\_direct | 1.00 | 0.653 | 0.467 | — |
| agreement\_check | 0.445 | 0.780 | 0.571 | — |
| confidence\_threshold | 0.990 | 0.657 | 0.472 | — |
| probe\_heuristic | 1.00 | 0.620 | 0.461 | — |
| prompt\_only\_abstain | 0.493 | 0.669 | 0.409 | 0.339 |
| **flat action-only (control)** | 0.516 | 0.816 | **0.610** | **0.506** |
| **flat four-head (structured)** | 0.596 | 0.833 | 0.568 | 0.396 |

**Key finding**: The structured four-head model *underperformed* the simpler action-only control on action accuracy (0.396 vs 0.506). The model learned vision informativeness well (0.850) but failed on text informativeness (0.509 ≈ chance) and pairwise relation (0.473). C4 abstain rate was only 0.341 (should approach 1.0).

### Root Causes Identified

1. **Feature deficiency**: Probe features were 3-dim (entropy, margin, variance) — purely unimodal uncertainty. Missing the most discriminative signal: do the two modalities actually *agree on an answer*?

2. **Flat multi-task architecture**: All four heads read from the same trunk independently. No information flows from informativeness predictions to relation, or from relation to action — despite the obvious semantic hierarchy.

3. **Loss imbalance**: Equal weight (`λ=1.0`) for all four tasks means 75% of gradient budget goes to intermediate noisy targets, hurting the primary action objective.

4. **Insufficient training**: 1 epoch with patience=1 is far too short for multi-task convergence.

5. **Lossy backbone representation**: `hidden_states` are the last Qwen layer truncated to `[:, :128]` out of 3,584 dims — only 3.6% of the actual representation. This is indefensible as a design choice in a paper.

---

## 4. Improvement Plan (Implemented)

### 4.1 Cross-Modal Agreement Features (`carm/models/features.py`)

New `extract_cross_modal_features(v_dist, t_dist, v_text, t_text)` → 5-dim `phi_cross`:

| Feature | Encodes |
|---|---|
| `agreement_binary` | Do predicted answers match? (1/0) |
| `dist_overlap` | Soft agreement: `sum(min(v_dist, t_dist))` |
| `v_max_prob` | Vision-only confidence |
| `t_max_prob` | Text-only confidence |
| `both_confident_disagree` | `v_max * t_max * (1 - agreement)` — the C4 signature |

This gives the model the signal it needs to distinguish C1 (agree) from C4 (both confident, disagree) from C2/C3 (asymmetric confidence) from C5 (both low confidence).

### 4.2 Cascade Architecture (`carm/experimental/model.py`: `CascadeCARMHeads`)

Replaces the flat parallel-head design with a hierarchical cascade:

```
Stage 1:  trunk(features) → vision_info_logits, text_info_logits
Stage 2:  [trunk_out | softmax(vision_info) | softmax(text_info)] → relation_logits
Stage 3:  [trunk_out | softmax(vision_info) | softmax(text_info) | softmax(relation)] → action_logits
```

The action head explicitly observes predicted informativeness and relation signals before deciding. This matches the semantic dependency: *know if each modality is informative → understand their relationship → decide the action*.

### 4.3 Distribution-Based Input (Recommended — `DistributionCARMHeads`)

**The most important architectural change.** Replaces truncated hidden states with the three answer distributions as primary inputs:

```
Input: [mm_dist(35), v_dist(35), t_dist(35), phi_cross(5)] = 110-dim
```

The `answer_dist` is Qwen's probability over the closed 35-token vocab (`{yes, no, 0–20, red, blue, ...}`). It is already computed and cached. It directly encodes what each modality predicts — making CARM a principled **distribution arbitrator** rather than a hidden-state classifier.

**Why this matters for the paper**: The claim becomes *"CARM learns to route based on modality-conditional answer distributions and their agreement structure."* This is verifiable, interpretable, and does not depend on an arbitrary truncation choice.

### 4.4 Training Improvements

- Loss weights: `λ_action=3.0`, `λ_relation=1.5`, `λ_info=1.0` — action is the primary objective
- Training budget: `epochs=10`, `patience=3`

### 4.5 New Configs

| Config | Model type | Purpose |
|---|---|---|
| `cascade_carm_preflight.yaml` | cascade | 15-example pipeline validation |
| `cascade_carm_10pct.yaml` | cascade | Full 10% training run |
| `distribution_carm_preflight.yaml` | distribution | 15-example pipeline validation |
| `distribution_carm_10pct.yaml` | **distribution** | **Recommended full run** |

---

## 5. Current Status

- `RUN-CTRL-0002`: cancelled (was running on incorrect dataset version)
- All code changes tested and passing
- Dataset refreshed locally (`carm_vqa_5way_10pct_protocol_family_seed7`, updated 2026-03-16)
- **Next action**: run `distribution_carm_preflight` to validate pipeline, then `distribution_carm_10pct` for the main results

---

## 6. Future Directions

### 6.1 Near-Term (Paper Completion)

**Ablation table** — this is required for any top-venue submission. Need to compare:

| Model | Inputs | Architecture |
|---|---|---|
| flat, action-only | hidden states | single head |
| flat, four-head | hidden states | parallel heads |
| cascade | hidden states + phi\_cross | cascade |
| **distribution (cascade)** | **answer dists + phi\_cross** | **cascade** |

Each row isolates one change. The paper story: structured inputs and hierarchical prediction both matter, and the right inputs matter more than the architecture.

**OOD generalization**: The current splits include held-out conflict families and corruption severities. Running the best model on OOD splits is required for any generalization claim.

**Risk-coverage analysis**: Plot task-success risk-coverage curves for cascade/distribution CARM vs. baselines. The learned model should dominate the non-learned baselines at the same coverage level.

### 6.2 Medium-Term Improvements

**Richer cross-modal features**: The current `phi_cross` uses text-normalized answer matching. A stronger version would use family-specific canonicalization (e.g., integer comparison for count questions) and soft semantic similarity for open-ended answers.

**Auxiliary contrastive objective**: Add a contrastive loss that pushes C1 and C4 apart in representation space, since both have both modalities informative but differ only in agreement. This is the hardest case for the current model.

**Calibration analysis**: CARM's action confidence should be calibrated — when it predicts `abstain` with high confidence, it should be right. Binned calibration plots and ECE metrics would strengthen the paper's calibration claims.

**Multi-backbone evaluation**: Run the same CARM heads on a second backbone (e.g., LLaVA-1.5 or InstructBLIP) to verify the "model-agnostic" claim. This is a key differentiator from fine-tuning approaches.

### 6.3 Longer-Term Research Directions

**Uncertainty-aware arbitration**: The current phi\_cross encodes confidence but treats it as a fixed feature. A more principled approach: model CARM as a Bayesian arbitrator that explicitly reasons about reliability under uncertainty, similar to sensor fusion in robotics.

**Mechanistic grounding**: Recent work shows that multimodal routing happens through specific attention heads in deeper layers. Extracting features from these heads (rather than the full last layer) could yield a more targeted representation for CARM.

**Extending beyond closed vocabulary**: The distribution-based design requires a fixed closed vocabulary. For open-ended questions, one approach is to use an embedding-based similarity score between generated answers rather than a discrete vocab distribution.

**Online adaptation**: CARM is currently trained offline. An online adaptation variant could update CARM's heads continuously as new conflict patterns emerge, without touching the frozen backbone.

---

## 7. Files Added / Modified

```
MODIFIED:
  carm/models/features.py              → +extract_cross_modal_features (5-dim phi_cross)
  carm/experimental/model.py           → +CascadeCARMHeads, +DistributionCARMHeads
  carm/experimental/training.py        → supports all three model types, passes phi_cross
  carm/experimental/evaluation.py      → StructuredCARMPredictor handles all model types
  scripts/run_experimental_small_data.py → model.type: flat|cascade|distribution

NEW:
  configs/cascade_carm_preflight.yaml
  configs/cascade_carm_10pct.yaml
  configs/distribution_carm_preflight.yaml
  configs/distribution_carm_10pct.yaml
  RESEARCH_NARRATIVE.md
  AUTO_REVIEW.md
  REVIEW_STATE.json
  PROJECT_BRIEF.md
```

---

## 8. Key References

- Hua et al., 2025 — modality dominance and unstable evidence use in MLLMs
- Popordanoska et al., 2025 (CLASH) — cross-modal contradiction detection benchmark
- Zhang et al., 2025 — modality-bias decomposition into uncertainty and preference
- Wen et al., 2025 — abstention as a calibrated safety mechanism
- Zhang et al., 2026 — instruction tokens as structural anchors for multimodal routing
