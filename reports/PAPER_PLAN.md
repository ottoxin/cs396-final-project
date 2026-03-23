# PAPER_PLAN.md — CARM Final Paper

Generated: 2026-03-16
Target venue: NeurIPS 2024 format (class project)
Page target: 8-9 pages body + references

## Title
**When Senses Disagree: Calibrated Arbitration for Conflicting Multimodal Reasoning**
(keep existing title — it's good)

## Claims–Evidence Matrix

| Claim | Evidence | Section |
|-------|---------|---------|
| CARM achieves +27.2% task success over best non-learned baseline | cascade test TaskSuccess=0.734 vs agreement_check=0.577 | Sec 5 Table 1 |
| Hidden-state input dramatically outperforms distribution input | cascade 0.734 vs dist v1 0.599 (+13.5pp); vision_info_acc 0.883 vs 0.648 | Sec 6 ablation |
| Class-weight rebalancing helps but is insufficient | dist v2 0.632 vs dist v1 0.599 vs cascade 0.734 | Sec 6 ablation |
| C4/C5 abstention reliably learned | C4 TaskSuccess=0.756, Abstain%=75.6%; C5=0.791, Abstain%=79.1% | Sec 5 per-category table |
| C2 (trust_vision) collapse fixed by cascade | dist v1 C2=0.237 → cascade C2=0.807 | Sec 5 per-category |
| C3 (trust_text) remains the hardest case | all models struggle: dist v1=0.000, dist v2=0.274, cascade=0.511 | Sec 6 failure analysis |
| Cascade achieves high accuracy-on-answered (0.847) | AccOnAnswered=0.847 vs backbone_direct=0.641 | Sec 5 |

## Section Structure

### 1. Abstract (~150 words)
- Problem: MLLM brittleness under modal conflict
- Method: CARM lightweight arbitration layer with cascade heads
- Key result: 73.4% task success, +27.2% over best non-learned baseline
- Per-category: C1/C2 solved, C4/C5 abstention robust, C3 partial

### 2. Introduction (~1 page)
- Hook: MLLMs fail under modal conflict
- Gap: no explicit arbitration mechanism
- Contribution: CARM with 5-category protocol + CascadeCARMHeads
- Preview main results (73.4%, per-category)
- Roadmap

### 3. Background & Related Work (~1 page)
- Keep existing content largely (well-written)
- Add: connection to abstention literature, CLASH benchmark
- Trim timetable/deliverables/team sections

### 4. Problem Formulation: Five-Category Conflict Protocol (~0.75 page)
- Define C1-C5 formally
- Oracle action mapping
- Dataset: 44,982 rows from VQAv2/COCO, 10% experimental subset
- Evaluation metrics: task_success_revised, action_accuracy, coverage, AccOnAnswered

### 5. CARM Architecture (~1.5 pages)
- Overview: frozen backbone + lightweight cascade heads
- Unimodal probes: phi_v (3-dim), phi_t (3-dim)
- Cross-modal features: phi_cross (5-dim) — NEW, explain each
- CascadeCARMHeads: 3-stage cascade (info → relation → action)
  - Stage 1: vision_info_head, text_info_head (2-way each)
  - Stage 2: relation_head conditioned on info logits
  - Stage 3: action_head conditioned on info + relation logits
- Training objective: multi-task cascade loss

### 6. Experiments (~2.5 pages)
#### 6.1 Setup
- Dataset, splits (train=31484, val=6743, test=6755), 10% subset
- Backbone: Qwen2.5-VL-7B (frozen)
- Baselines: backbone_direct, agreement_check, confidence_threshold, probe_heuristic, prompt_only_abstain

#### 6.2 Main Results (Table 1 — full baseline comparison)
- All predictors, all metrics
- Cascade CARM vs all baselines

#### 6.3 Per-Category Results (Table 2)
- C1-C5 TaskSuccess, ActionAcc, Abstain% for cascade CARM
- Key: C2 fixed (0.807), C3 partial (0.511), C4/C5 robust

#### 6.4 Architecture Ablation (Table 3)
- Dist v1 | Dist v2 | Cascade — overall + per-category
- Shows: input representation > loss weighting

### 7. Analysis & Discussion (~1 page)
- Why hidden states win: modality quality encoding
- C3/C5 confusability: phi_cross limitation (low v_max_prob present in both C3 and C5)
- Contradiction_error_rate interpretation (contextualize 0.852)
- Coverage-precision tradeoff discussion

### 8. Limitations (~0.5 page)
- Single seed, 10% subset
- C3 gap (40% false abstention)
- VQAv2/COCO only (English, factual)
- No OOD generalization test

### 9. Conclusion (~0.25 page)
- Summary of contributions
- Cascade architecture + explicit arbitration = strong baseline for future work

## Figure Plan

### Figure 1: CARM Architecture Diagram (MANUAL — skip for now, describe in text)
- Input: image, text, question
- Frozen backbone → hidden states
- Unimodal probes → phi_v, phi_t, phi_cross
- Cascade: info → relation → action
- Action-conditioned generation

### Table 1: Main Baseline Comparison (AUTO-GENERATE as LaTeX table)
Columns: Method | Coverage | AccOnAnswered | TaskSuccess | ActionAcc | Delta
Rows: 6 baselines + 3 CARM variants

### Table 2: Per-Category CARM Results (AUTO-GENERATE as LaTeX table)
Rows: C1-C5, Overall
Columns: n | ActionAcc | TaskSuccess | Abstain%

### Table 3: Architecture Ablation (AUTO-GENERATE)
Rows: Dist v1, Dist v2, Cascade
Columns: Architecture | Input | Overall TaskSuccess | C2 | C3 | C4 | C5

### Figure 2: Per-category bar chart (OPTIONAL — simple, can do in matplotlib)

## Citation Plan
Keep existing citations. Add:
- VQAv2 / COCO dataset citation
- Qwen2.5-VL citation
- CLASH (Popordanoska 2025) — already cited
- Abstention/selective prediction (Wen 2025) — already cited

## Files to Create/Modify
- `writeup/carm_proposal.tex` → overwrite with full results paper
- No separate sections files needed (single-file is fine for 9-page paper)
