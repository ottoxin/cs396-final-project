# CARM Top-Tier Plan (VQAv2 + COCO Captions)

## 0) Claim (paper-level)
We frame multimodal conflict handling as separable supervised decisions:
1) conflict type prediction,
2) question-conditioned modality reliability estimation (`r_v`, `r_t`),
3) action selection (`trust_vision`, `trust_text`, `require_agreement`, `abstain`),
4) action-conditioned generation.

Main paper claim: this decomposition improves OOD arbitration and calibrated selective prediction under disagreement, beyond obvious routing heuristics.

### 0.1 Overview plan by version
- **v1 (current locked protocol. We may end here first for purpose for the class.):** Build Conflict Suite on VQAv2+COCO with families `existence/count/attribute_color`, operators `swap_easy/swap_hard/text_edit/vision_corrupt`, OOD-family + OOD-severity evaluation, and full baseline/reporting pipeline.
- **v2 (harder single-dataset protocol):** Extend conflict stress testing within the same base dataset (for example, optional both-corrupted defaults, stricter hard-swap/OOD-hard-swap emphasis, and tighter integrity constraints) while preserving v1 comparability.
- **v3 (cross-dataset generalization):** Expand to multi-dataset evaluation (for example adding GQA and one instruction-following set), then validate arbitration/calibration transfer across datasets and backbone families.

---

## 1) Scope decisions (locked for v1)

### Backbones (2)
- Backbone A: **Qwen2.5-VL-7B-Instruct**
- Backbone B: **LLaVA-NeXT (Llama 3 8B, HF checkpoint)**

### Dataset (1 for current cycle)
- Primary: **VQAv2 questions + COCO captions** (caption as text modality)
- Explicit future extension (not in v1 core): add **GQA** and one instruction-following dataset

---

## 2) Benchmark: Conflict Suite v1 on VQAv2+COCO

### 2.1 Base record
Each base example contains:
- image `I`
- caption `T` (COCO)
- question `q` (VQAv2)
- gold answer `y`

### 2.2 Question families (clean first)
- `existence` (yes/no)
- `count`
- `attribute_color` (closed-vocabulary color; include in v1)

### 2.3 Conflict operators (v1)
- `swap_easy`: random caption swap
- `swap_hard`: within-bucket swap to reduce topic shortcuts
- `text_edit`: family-aligned edits (count/negation/color)
- `vision_corrupt`: one corruption type first, severities `{1,2,3}`

Optional extension: both-corrupted or ambiguous variants for abstention stress-testing.

### 2.4 Metadata schema (must exist per generated example)
- `base_id`, `variant_id`
- `family`
- `operator`
- `corrupt_modality` (`none|text|vision|both`)
- `severity`
- `heldout_family_flag`, `heldout_severity_flag`, `hard_swap_flag`
- ground-truth answer `y`
- reference action `a*`

---

## 3) Labels and reliability supervision

### 3.1 Reference action labels (`a*`, deterministic)
- text corrupted, vision clean -> `trust_vision`
- vision corrupted, text clean -> `trust_text`
- none corrupted and consistent -> `require_agreement`
- both corrupted or ambiguous -> `abstain`

### 3.2 Reliability targets
Targets are question-conditioned and derived from construction metadata.

Validation requirements:
- monotonicity with severity
- calibration vs empirical correctness (binned, stratified by family)

### 3.3 Collapse prevention
Use counterfactual regularization `L_cf` to avoid degenerate policies (for example, always trusting one modality).

Mandatory ablation: with vs without `L_cf`.

---

## 4) Method: CARM

### 4.1 Multi-pass setup
- pass 1: multimodal `(I, T, q)`
- pass 2: vision-only `(I, q)`
- pass 3: text-only `(T, q)`

Cache pass outputs/features for training and evaluation.

### 4.2 Features
- anchored hidden states from backbone
- probe features from unimodal passes (entropy, top-2 margin, logit gap)
- optional disagreement indicators (vision answer vs text answer mismatch)

### 4.3 Heads
- conflict head: `p(c | ·)`
- reliability head: `r_v, r_t`
- action head: `p(a | c, r_v, r_t, features)`

### 4.4 Action-conditioned generation
- `trust_vision`: generate from `(I, q)`
- `trust_text`: generate from `(T, q)`
- `require_agreement`: answer only when unimodal answers match; else abstain
- `abstain`: return abstention output token/string

---

## 5) OOD evaluation (headline)

### 5.1 Required OOD axes
1) **OOD-family**: hold out one family from training
2) **OOD-severity**: train on severities `{1,2}`, test on `{3}`

### 5.2 Optional OOD axis
3) **OOD-hard-swap**: train on easy swaps, test on hard swaps

### 5.3 Split reporting
- Train/Val/Test-ID/Test-OOD-family/Test-OOD-severity (+ optional Test-OOD-hard-swap)
- Split manifests and integrity checks are required artifacts

---

## 6) Baselines (must clear strong heuristics)

Per backbone:
1) direct backbone (full input)
2) verifier prompting -> answer
3) uncertainty-threshold abstention
4) two-pass self-consistency router (vision-only vs text-only, abstain on disagreement)
5) probe-only heuristic router (entropy/margin)
6) optional learned router on probe features

---

## 7) Metrics and diagnostics

### 7.1 Main metrics
- answer accuracy (consistent vs conflict; per family)
- action accuracy
- conflict-type macro F1
- risk-coverage curves (ID and OOD)
- reliability monotonicity and calibration

### 7.2 Required diagnostics/ablations
- action distribution entropy (collapse check)
- remove probe features
- remove `L_cf`
- remove abstention
- pass-budget: 1 vs 2 vs 3 passes

---

## 8) Reproducibility and release artifacts
- conflict-suite generator scripts and operator definitions
- split files/manifests
- unified train/eval pipeline
- cached probe output format/scripts
- configs + seed control + environment capture
- figure/table generation scripts
- qualitative casebook (10-20 examples)

---

## 9) Execution phases

### Phase A: Benchmark + baselines
- build suite v1 with deterministic templates and normalization
- implement all baselines including two-pass router
- produce ID/OOD tables on Backbone A

### Phase B: CARM + ablations
- train heads with `L_cf`
- run required ablations
- produce risk-coverage and reliability plots
- replicate on Backbone B

### Phase C: Consolidation + writing
- error analysis and qualitative casebook
- finalize figures/tables and claim language
- document future extension to multi-dataset evaluation

---

## 10) Defaults frozen for this execution cycle

These defaults are active unless overridden in config/CLI:

- OOD-family holdout: `attribute_color`
- included families: yes/no + count + color
- first vision corruption type: `occlusion`
- variants per base sample: `swap_easy`, `swap_hard`, `text_edit`, `vision_corrupt`
- abstention output token/string: `<ABSTAIN>`

---

## 11) Current codebase snapshot and gap-to-claim

Implemented now:
- end-to-end scaffold (data schema, generation, labels, train/eval scripts, tests)
- deterministic mock backbone path for CPU-local verification

Still required to match final paper claim:
- real Qwen2.5-VL and LLaVA-NeXT adapters
- production multi-pass caching pipeline
- final operator set and metadata alignment with v1 spec
- missing two-pass self-consistency baseline in baseline runner
- complete OOD split generation and reporting hooks

---

## 12) Immediate engineering priorities (next milestones)

1) encode v1 defaults in configs and generator CLI paths
2) finalize Conflict Suite v1 operators + metadata
3) implement two-pass self-consistency baseline with abstention
4) finalize OOD-family and OOD-severity split generation/reporting
5) add risk-coverage + calibration artifacts to evaluator outputs
6) integrate real backbone adapters while preserving mock mode for tests

---

## 13) Decision hygiene and anti-drift rule

- `PLAN.md` is the single source of truth for research decisions and frozen defaults.
- Any change to families, operators, OOD protocol, backbones, or acceptance criteria must update this file in the same commit as code/config changes.
