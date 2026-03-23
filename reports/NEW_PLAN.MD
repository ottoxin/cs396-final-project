# Updated Plan for Next Step

HF numbering correction used throughout the active repo state:

* `C1 = consistent / require_agreement`
* `C2 = trust_vision / text uninformative`
* `C3 = trust_text / vision uninformative`
* `C4 = contradictory / abstain`
* `C5 = both weak / abstain`

## Objective

Use the **current CARM repo** to build a **small-data experimental path** that tests a revised structured design for multimodal conflict handling.

Great. Since you already prepared the **10% data with the new fields**, the next step should be a clean **end-to-end experimental plan** that answers four things:

1. do the new labels behave as intended
2. do simple baselines already separate the cases
3. does structured training improve over the old action-only head
4. where does the system fail and why

Below is the plan I would follow.

---

# End-to-end plan after preparing the 10% data

## 0. Overall goal of this stage

This stage is **not** about final benchmarking.
It is about validating the revised formulation:

* `vision_supported_target`
* `text_supported_target`
* `vision_info_state`
* `text_info_state`
* `pairwise_relation`
* `joint_action`
* `joint_target`

and testing whether the current repo can support a structured CARM with:

* modality informativeness
* pairwise consistency
* joint action

The success criterion is:

* clean pipeline
* interpretable outputs
* sensible diagnostics
* evidence that structured supervision is worth keeping

---

# 1. First: data audit and sanity verification

Before running any baseline or training, produce a **data sanity report**.

## 1.1 Check field completeness

For the 10% dataset, report counts for:

* `vision_supported_target` available / missing
* `text_supported_target` available / missing
* `vision_info_state`
* `text_info_state`
* `pairwise_relation`
* `joint_action`
* `joint_target`

This should be broken down by **C1–C5**.

## 1.2 Check category-to-label mapping

Create one table like:

| Cat |   vision_info |     text_info | relation      | joint_action      | joint_target |
| --- | ------------: | ------------: | ------------- | ----------------- | ------------ |
| C1  |   informative |   informative | consistent    | require_agreement | gold         |
| C2  |   informative | uninformative | asymmetric    | trust_vision      | gold         |
| C3  | uninformative |   informative | asymmetric    | trust_text        | gold         |
| C4  |   informative |   informative | contradictory | abstain           | `<ABSTAIN>`  |
| C5  | uninformative | uninformative | both_weak     | abstain           | `<ABSTAIN>`  |

Verify the prepared 10% data actually matches this mapping.

## 1.3 Save a small example sheet

For each category, save 10 representative rows with:

* question
* gold answer
* vision_supported_target
* text_supported_target
* all new labels

This becomes the first debugging artifact.

---

# 2. Evaluation setup: split the experiment into 3 panels

I would structure the whole experiment in **three panels**.

## Panel A. Answer-forcing behavior

Purpose:

* characterize what happens when the model must answer

Methods:

* `backbone_direct`

This is **not** a selective policy baseline. It is the forced-answer reference.

## Panel B. Non-learned diagnostic/selective baselines

Purpose:

* test whether simple signals already encode useful structure

Methods:

* `prompt_only_abstain`
* `agreement_check`
* `confidence_threshold`
* optionally `probe_heuristic` only if trivial to include

## Panel C. Learned models

Purpose:

* compare the old head vs new structured head

Methods:

* old repo `action-only CARM`
* new structured CARM:

  * informativeness + consistency + action

This setup gives you a very clean narrative.

---

# 3. Baselines: what to run and how to evaluate them

## 3.1 Backbone direct

Role:

* answer-forcing baseline

Report:

* overall answer accuracy
* per-category answer accuracy
* error rate on:

  * contradiction (C4)
  * asymmetry (C2/C3)
  * both weak (C5)

Do **not** center task success here, because for C4/C5 it is structurally mismatched.

## 3.2 Prompt-only abstain

This is important.

### Tier 1 first

Use a strict prompt that allows:

* answer
* or `ABSTAIN`

Same backbone, no trained head.

Purpose:

* check whether abstention is already latent when explicitly allowed

Report:

* coverage
* accuracy on answered
* task success
* abstain rate by category

### Tier 2 later, only if stable

Policy-token output:

* `TRUST_VISION`
* `TRUST_TEXT`
* `REQUIRE_AGREEMENT`
* `ABSTAIN`

Do not start here unless Tier 1 is stable.

## 3.3 Agreement check

Role:

* contradiction detector by behavior

Rule:

* if vision/text answers match, answer
* else abstain

Report:

* task success
* coverage
* accuracy on answered
* abstain rate by category
* especially how often it abstains on C4 vs C1

## 3.4 Confidence threshold

Role:

* uncertainty-only diagnostic baseline

Rule:

* answer if multimodal confidence exceeds threshold
* else abstain

Use a **small fixed threshold grid** on val only, not extensive tuning.

Report:

* best threshold on val
* test metrics
* confidence distribution by category

## 3.5 Optional probe heuristic

Only include if it is already easy.
It is lower priority now.

---

# 4. Diagnostics: justify the features and revised design

This part is very important because it tells the story of **why** the structured model is needed.

## 4.1 Diagnostics to run before training

### A. Unimodal agreement by category

For each category, compute:

* `% vision/text agree`
* `% both correct`
* `% only vision correct`
* `% only text correct`
* `% neither correct`

Expected patterns:

* C1: high agreement
* C2: high disagreement, both may still be individually valid vs their own targets
* C3: image stronger than text
* C4: text stronger than image
* C5: weak / unstable

This directly motivates:

* consistency feature
* modality informativeness

### B. Confidence / entropy by category

For each of:

* multimodal entropy
* vision entropy
* text entropy
* margin

plot distributions by category.

You want to see whether:

* C2 differs from C1
* C5 differs from C1/C2
* C3/C4 are asymmetric in uncertainty

This justifies uncertainty features.

### C. Feature separability for the new labels

Using the new 10% labels, compare features across:

* `pairwise_relation = consistent`
* `contradictory`
* `asymmetric`
* `both_weak`

and separately by:

* `vision_info_state`
* `text_info_state`

Use simple plots:

* boxplots
* violin plots
* pairwise scatter plots
* PCA/UMAP only if easy and honest

### D. Candidate-answer behavior for C2

This is one of the most important diagnostics.

For C2, explicitly show:

* vision answer vs `vision_supported_target`
* text answer vs `text_supported_target`
* multimodal answer
* whether direct multimodal collapses to one side

This is the clearest justification for why C4 should be **joint abstain** rather than “trust one side.”

---

# 5. Visualizations to include

I would plan these figures from the start.

## Figure 1. Label schema summary

A compact diagram or table:

* C1–C5
* per-modality informativeness
* pairwise relation
* joint action

This explains the whole benchmark revision.

## Figure 2. Unimodal agreement behavior by category

Stacked bar chart:

* agree
* disagree
* only vision correct
* only text correct
* neither

This motivates consistency modeling.

## Figure 3. Confidence / entropy by category

Boxplots or violin plots for:

* multimodal entropy
* vision entropy
* text entropy

This motivates uncertainty features.

## Figure 4. Baseline comparison

Two panels:

### Panel A

Answer accuracy only for `backbone_direct`

### Panel B

Coverage / task success / accuracy-on-answered for:

* prompt_only_abstain
* agreement_check
* confidence_threshold
* structured CARM

## Figure 5. Structured CARM architecture

Use the revised figure with:

* candidate answers
* informativeness head
* consistency head
* action head
* router

## Figure 6. Per-category learned-model results

Grouped bars for:

* old action-only head
* new structured head

across C1–C5.

## Figure 7. Confusion matrices

For the learned model:

* pairwise relation confusion
* action confusion

## Figure 8. Failure diagnostic examples

A few example cases:

* C4 contradiction
* C3 asymmetry
* C5 both weak

showing:

* unimodal outputs
* predicted informativeness
* predicted consistency
* predicted action
* gold action

---

# 6. Training plan

## 6.1 Start with two learned models

### Model A: old action-only head

This is your closest baseline inside the learned family.

Input:

* current feature vector

Output:

* `joint_action`

This is the control condition.

### Model B: structured CARM

Input:

* same feature vector

Outputs:

* `vision_info_state`
* `text_info_state`
* `pairwise_relation`
* `joint_action`

This is the main new model.

This comparison is crucial because it tells you whether the new supervision matters.

---

## 6.2 Input features

Use current repo features first.

### Core features

* pooled hidden-state / anchor representation
* multimodal uncertainty features
* vision-only uncertainty features
* text-only uncertainty features
* answer agreement indicator
* maybe answer identity overlap features if already available

Do not add many new feature-engineering tricks yet.

The point is to test whether the **new labels and structure** help, not whether a giant feature soup helps.

---

## 6.3 Head structure

I would start with:

* one shared MLP trunk
* separate heads

### Head 1

Vision informativeness binary classification

### Head 2

Text informativeness binary classification

### Head 3

Pairwise relation 4-way classification

### Head 4

Joint action 4-way classification

You can later merge the two informativeness heads into one joint head, but keep it simple first.

---

## 6.4 Losses

Use masked multi-task loss.

[
\mathcal{L} =
\lambda_v \mathcal{L}*{v_info} +
\lambda_t \mathcal{L}*{t_info} +
\lambda_r \mathcal{L}*{relation} +
\lambda_a \mathcal{L}*{action}
]

Start with equal weights:

* `lambda_v = 1`
* `lambda_t = 1`
* `lambda_r = 1`
* `lambda_a = 1`

Only rebalance if one task clearly dominates.

### Masking

* if any target missing, that task is masked for that example
* log counts per task

---

## 6.5 Training schedule

Keep it modest.

### Stage 1

Tiny sanity run

* 1–3 epochs
* verify losses, outputs, saved artifacts

### Stage 2

Main 10% run

* small fixed train/val/test split
* early stopping on validation action metric or task success

### Stage 3

Ablations
Run:

* action-only
* action + relation
* info + action
* info + relation + action

This tells you what the new structure is actually doing.

---

# 7. Evaluation plan

## 7.1 Core metrics

### For Panel A: backbone_direct

Report:

* answer accuracy
* per-category accuracy
* contradiction error
* asymmetry error
* both-weak error

### For Panel B/C: selective and learned models

Report:

* coverage
* accuracy on answered
* task success
* action accuracy
* macro-F1 for action
* abstain rate
* per-category task success

## 7.2 Additional structured metrics for the new model

Report:

* vision informativeness accuracy
* text informativeness accuracy
* pairwise relation accuracy / macro-F1
* relation confusion matrix
* action confusion matrix

These are essential. Otherwise you cannot tell whether the structure is really learned.

## 7.3 C4-specific evaluation

This deserves a separate subsection.

For C4, report:

* vision-only accuracy against `vision_supported_target`
* text-only accuracy against `text_supported_target`
* multimodal abstain rate
* multimodal non-abstain error rate

This directly tests your key semantic change.

---

# 8. Diagnostics after training

## 8.1 Per-example prediction artifact

For every example, save:

* gold and predicted:

  * vision_info_state
  * text_info_state
  * pairwise_relation
  * joint_action
* vision/text/mm answers
* entropies / margins
* task success
* whether abstained

This is the central debug file.

## 8.2 Failure buckets

Create `failure_diagnostics.csv` with buckets like:

### A. C4 contradiction collapse

Model should abstain but trusts one side

### B. False contradiction

Model predicts contradiction where case is asymmetric or consistent

### C. Informativeness error

Predicts informative when modality is supposed to be uninformative

### D. Wrong abstention

Abstains in C2/C3 when one modality should be trusted

### E. Overconfident wrong answer

Important for prompt-only and direct baselines

---

# 9. Suggested analysis questions

Once results are in, answer these in order.

## Q1

Do the new labels behave sensibly in the 10% set?

## Q2

Do simple baselines already separate contradiction from irrelevance?

## Q3

Does the structured model improve over the old action-only head?

## Q4

Which intermediate target helps most:

* informativeness
* relation
* both

## Q5

Does the structured model especially improve C4 and C5?

## Q6

Are current features sufficient to distinguish:

* contradictory
* asymmetric
* both_weak

If not, that is a useful negative result and points to the next architecture step.

---

# 10. Concrete execution order

Here is the order I would actually run things.

## Step 1

Data sanity audit and example sheet

## Step 2

Run `backbone_direct` on 10% and compute Panel A metrics

## Step 3

Run `agreement_check` and `confidence_threshold`

## Step 4

Add `prompt_only_abstain` Tier 1

## Step 5

Generate diagnostics plots from baseline outputs:

* agreement by category
* entropy by category
* C4-specific behavior

## Step 6

Train old action-only head on 10%

## Step 7

Train structured CARM on 10%

## Step 8

Run ablations:

* action-only
* relation+action
* info+action
* info+relation+action

## Step 9

Produce confusion matrices and failure diagnostics

## Step 10

Write a short conclusion answering:

* what worked
* what failed
* what labels/features are insufficient
* what the next architecture step should be

---

# 11. What counts as success at this stage

This stage is successful if you can show:

* the new labels are coherent and derivable
* C2 is now handled with the intended semantics
* simple diagnostics already justify informativeness/consistency features
* the structured model is trainable and interpretable
* you can clearly tell whether the new supervision helps

It does **not** require beating every baseline yet.

---

# 12. The final story you want from this 10% stage

Ideally, by the end of this stage you want to be able to say:

1. the revised dataset semantics are clean
2. contradiction and irrelevance are meaningfully different
3. backbone-derived signals partially reveal these differences
4. a structured arbitration head can learn them more explicitly than the old action-only head
5. C4 is now modeled correctly as “both informative but contradictory, therefore joint abstain”
