# Conflict Suite v1 Data Construction Note

## Dataset Construction

### Raw data gathering
Conflict Suite v1 is constructed from official VQAv2 question-answer annotations and COCO captions, with each example anchored to a single COCO image. The construction objective is to create a high-validity benchmark for multimodal conflict handling under controlled perturbations while preserving deterministic supervision and reproducible splits. In the current protocol, we retain three question families that support reliable normalization and agreement checks: `existence`, `count`, and `attribute_color`. Detailed stage-by-stage specifications are provided in Appendix A1-A5.

For each VQAv2 question, we infer family membership from question prefixes and then normalize the gold answer under family-specific rules. Existence answers are normalized to `yes` or `no`, count answers are normalized to integer strings, and color answers are restricted to a fixed closed vocabulary. Any question that does not map to one of the selected families, or any answer that cannot be normalized under the active family rule, is excluded before variant generation (Appendix A1).

After normalization, we enforce caption consistency. Given all COCO captions for the image tied to a question, we retain the first caption that is compatible with the normalized answer under family-specific support rules. If no caption satisfies the rule, the example is dropped with a consistency-filter failure reason. This step produces clean base examples where the retained text evidence is compatible with the answer used for supervision (Appendix A2).

From each retained base example, we generate Conflict Suite v1 variants with deterministic metadata and oracle actions. The emitted set contains one clean variant, two caption-swap variants (`swap_easy` and `swap_hard`), one family-aligned text-edit variant, and three vision-corruption variants with severities 1, 2, and 3. The hard-swap operator uses constrained donor selection and falls back to easy-swap donor text when no valid hard donor is available, while preserving the `SWAP_HARD` operator tag and exposing fallback status through `hard_swap_flag` (Appendix A3). Oracle-action mapping rules are summarized in Appendix A4.

Split assignment is performed at source-image level to prevent leakage across variants derived from the same image. Base splits are assigned with deterministic seeded shuffling and default 70/15/15 source ratios for train/val/test-ID, then overwritten by protocol-level OOD rules for held-out family and held-out severity. The generated manifest records counts, hashes, configuration values, and integrity checks to support exact regeneration (Appendix A5).

In the current official-data build with consistency filtering enabled, the base-construction stage processed 658,111 candidate VQAv2 question-answer records and retained 184,590 clean base examples (28.0%), while filtering out 473,521 records (72.0%). The filtered set consists of 308,623 records removed by family gating, 22,060 removed by answer-normalization failure, and 142,838 removed by caption-consistency failure. The retained family composition is 150,582 existence examples, 15,207 count examples, and 18,801 attribute-color examples. These counts refer to an upstream clean-base filtering stage and must not be conflated with the current released refined corpus size.

### CARM Supervision Refinement (Five-Category Protocol)

The project objective is to train CARM to choose among four entropy-conditioned actions: `ABSTAIN`, `TRUST_VISION`, `TRUST_TEXT`, and `REQUIRE_AGREEMENT` (with abstention on disagreement). Deterministic supervision is provided through a fixed five-category protocol with category-to-action mapping C1 -> `REQUIRE_AGREEMENT`, C2/C5 -> `ABSTAIN`, C3 -> `TRUST_VISION`, and C4 -> `TRUST_TEXT`. The formal action mapping is aligned with Appendix A4.

For planning and release accounting, we distinguish the nominal balanced target from the realized public dataset. The nominal refined target is `45,000` examples under equal family-category balancing, while the current HF release used by the active workflow contains `44,982` rows (`nbso/carm-vqa-5way`) after realized filtering and moderation outcomes. Detailed category definitions, balancing tables, split-allocation arithmetic, and caption-edit workload accounting are moved to Appendix A6.

### Training Objective

We train CARM as a lightweight four-way policy learner on top of a frozen multimodal backbone. For each image-caption-question triple, the backbone provides pooled multimodal hidden states, and unimodal probe passes over the vision and text channels produce family-specific answer distributions. We summarize these unimodal distributions using compact uncertainty features, including entropy and top-1 margin, and combine them with the pooled backbone representation as input to CARM. The module outputs a distribution over four actions: `trust_vision`, `trust_text`, `require_agreement`, and `abstain`.

The primary supervision signal is the oracle action implied by the benchmark construction protocol. Let $a_i^*$ denote the oracle action for example $i$, and let $p_\theta(a \mid x_i)$ denote the action distribution predicted by CARM for input $x_i$. We train the main model with standard cross-entropy:

$$
\mathcal{L}_{\text{action}}
=
-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(a_i^* \mid x_i).
$$

This objective is aligned with the role of CARM at inference time. Rather than predicting the final answer token directly, CARM is trained to choose the correct arbitration rule for combining, selecting, or rejecting modality-specific evidence.

The action `require_agreement` is treated as a distinct policy label during training. At inference time, this action induces a conditional rule: the system returns an answer only when the vision-only and text-only probes agree under family-specific normalization; otherwise, it abstains. This allows the model to distinguish cases in which agreement between modalities is itself the criterion for answering from cases in which one modality should be trusted directly or the system should abstain outright. Under the revised five-category protocol, this agreement label is reserved for the clean consistency case (C1), while contradiction cases (C2) are supervised directly as `abstain` and analyzed with separate unimodal-versus-multimodal diagnostics.

We freeze the backbone throughout training in order to isolate the contribution of the arbitration layer. This design keeps the learning problem focused on modality selection under conflict, rather than broad end-to-end adaptation of the underlying VLM. We optimize the trainable heads with Adam-style updates and use held-out decision quality, especially validation task success, as the primary model-selection criterion. Alongside task success, we report action accuracy, macro-F1, and selective-prediction metrics as secondary diagnostics.

In multi-head variants, we also study auxiliary supervision for intermediate structure, such as conflict type or modality reliability. In those variants, the total objective is

$$
\mathcal{L}
=
\mathcal{L}_{\text{action}}
+ \lambda_{\text{conf}}\mathcal{L}_{\text{conf}}
+ \lambda_{\text{rel}}\mathcal{L}_{\text{rel}}.
$$

Here $\lambda_{\text{conf}}$ and $\lambda_{\text{rel}}$ are tuned on the validation set. We treat these auxiliary losses as optional regularizers rather than core components of the method, and retain them only when they improve held-out decision quality.

## Baseline Evaluation

The updated evaluation protocol uses the same constructed dataset with deterministic `train`, `val`, and `test` (`test_id`) partitions, where `val` is reserved for baseline development and threshold selection, and `test` (`test_id`) is reserved for locked final reporting. Baseline comparison is restricted to four methods (`backbone_direct`, `agreement_check`, `confidence_threshold`, and `probe_heuristic`).

The Qwen backbone answer path is open-generation rather than closed-vocabulary next-token selection. For each question family, the model is prompted to answer in a constrained natural form (`yes/no`, a single integer, or a single color word), after which the generated text is canonicalized before evaluation. This change is critical for count questions, where closed-vocabulary selection produced invalid behavior in earlier runs.

Evaluation is record-centric: each baseline emits standardized per-example core outputs containing input/gold context, `final_answer`, `abstained`, `confidence`, `correct`, and `task_success`, from which all aggregate quantities are derived. When evaluating CARM, optional diagnostic fields such as `pred_action`, `pred_conflict_type`, `r_v`, `r_t`, `audit`, and the C2-specific fields `c2_vision_only_correct`, `c2_text_only_correct`, and `c2_multimodal_abstained` may also be appended, but baseline comparison does not depend on those extra fields. Under the finalized C2 contract, prepared rows may also store top-level `vision_supported_target` and `text_supported_target`; evaluation reads those fields first and falls back to legacy `metadata.c2_text_supported_answer` only when analyzing older rows. In the current caption-derived local export, `text_supported_target` is derived directly from the contradictory caption when the caption supports an answer, and otherwise left null with `metadata.text_supported_target_source = "missing_after_caption_rule"` so the coverage boundary remains explicit.

Metric choice follows the simplified baseline interface. Headline reporting uses raw `accuracy`, `coverage`, `accuracy_on_answered`, `task_success`, and selective-prediction summaries from the task-success risk-coverage curve (`Risk@target`, `AURC`). Category-level C1-C5 breakdowns remain required because aggregate metrics alone can hide protocol-specific failure modes. For flat baselines, task success is intentionally outcome-based: C1, C3, and C4 succeed only when the method answers correctly without abstaining, while C2 and C5 succeed only under abstention. Action-aware routing diagnostics remain optional analyses for CARM outputs rather than headline baseline metrics. For C2 specifically, we report secondary diagnostics for vision-only accuracy, text-only accuracy, and multimodal abstention rate rather than folding those behaviors into the headline benchmark score.

To keep the main text concise, formal metric definitions, mathematical expressions, confidence standardization, and the locked validation-to-test workflow are provided in Appendix B.

---

## Appendix A. Data Construction Details

### A1. Family Inference and Answer Normalization

Family inference is performed directly from normalized question text. Questions beginning with `is` or `are` are mapped to `existence`; questions beginning with `how many` are mapped to `count`; and questions beginning with `what color` are mapped to `attribute_color`. Questions outside these patterns are excluded from v1.

Answer normalization is family-dependent. Existence answers are collapsed into canonical `yes` and `no` forms, including accepted aliases such as `y`, `n`, `true`, and `false`. Count answers are accepted when they are either digit strings or number words with deterministic integer conversion. Attribute-color answers are accepted only when the normalized token appears in the configured closed color vocabulary. Any record failing normalization is removed prior to caption filtering.

### A2. Caption Consistency Filtering

For each candidate base record `(I, q, y)`, the pipeline scans all captions attached to image `I` and retains the first caption that satisfies `caption_supports_answer(q, family, y, caption)`. If no caption passes, the record is dropped with reason `consistency_filter_failed`. This first-pass retention rule is deterministic given fixed caption order from source files.

Support checks are family-specific. For `attribute_color`, the caption must contain the normalized color as a whole word match. For `count`, numeric evidence is extracted from both digits and recognized number words in the caption, and the normalized answer integer must appear in that set. For `existence`, subject-like tokens are extracted from the question after stopword filtering and compared against caption tokens; overlap is treated as support for `yes`, while non-overlap is treated as support for `no`. This design favors deterministic filtering over open-ended semantic inference.

### A3. Conflict Suite v1 Variant Generation

Each retained clean base example is expanded into a fixed operator set: `clean`, `swap_easy`, `swap_hard`, `text_edit`, and `vision_corrupt` at severities `1, 2, 3`. The clean variant is normalized to canonical defaults, including `corrupt_modality=none` and `severity=0`.

The `swap_easy` variant replaces caption text with donor text sampled from another base example. The `swap_hard` variant attempts constrained donor selection requiring same family, same answer-type bucket, different source image, and noun-token Jaccard overlap within the configured range (default `[0.2, 0.7]`). If no donor satisfies all constraints, the system falls back to easy donor text while keeping operator `SWAP_HARD` and setting `hard_swap_flag=false`.

The `text_edit` variant applies family-aligned textual perturbation. Existence edits flip or insert negation, count edits replace the first detectable number token (or inject a fallback numeric clause when no token is present), and color edits replace a detected color token with an alternative from the color vocabulary. The `vision_corrupt` variants emit severity-indexed vision corruption metadata with corruption type `occlusion` by default. The generation stage records recipe metadata for vision corruption and does not require materializing altered image files in-place.

### A4. Deterministic Oracle Action Assignment

Under the refined five-category protocol, oracle action labels are deterministic functions of category: C1 (clean text + clean image) maps to `REQUIRE_AGREEMENT`, C2 (`DIFFERENT` text + clean image) maps to `ABSTAIN`, C3 (`IRRELEVANT` text + clean image) maps to `TRUST_VISION`, C4 (clean text + irrelevant image) maps to `TRUST_TEXT`, and C5 (`IRRELEVANT` text + irrelevant image) maps to `ABSTAIN`. For backward compatibility with legacy v1 data lacking subtype/category metadata, the fallback policy remains modality-based (`none` -> `REQUIRE_AGREEMENT`, `text` -> `TRUST_VISION`, `vision` -> `TRUST_TEXT`, `both/ambiguous` -> `ABSTAIN`).

### A5. Split Assignment, OOD Overrides, and Manifests

Split construction begins with source-image-disjoint base assignment. Unique source image identifiers are shuffled with a fixed seed and partitioned with default 70/15/15 ratios into train, validation, and test-ID sources. Every variant derived from a source image inherits this base split before OOD overrides are applied.

OOD assignment then follows fixed priority: records in the held-out family are assigned to `TEST_OOD_FAMILY`; otherwise, corrupted records at or above held-out severity are assigned to `TEST_OOD_SEVERITY`; and optional hard-swap OOD assignment is applied when enabled and when `hard_swap_flag` is true. Final artifacts include split-wise counts, split hashes, configuration echoes, and integrity-validation outputs, enabling deterministic regeneration and leakage checks.

### A6. Five-Category Supervision, Balancing, and Workload Accounting

The five-category supervision protocol specifies the text and image condition pair for each category and fixes the oracle action accordingly. This mapping is the operational interface used for deterministic action supervision in both baseline and CARM training workflows.

| Category | Text condition | Image condition | Oracle action | Expected behavior | LLM caption edit |
| --- | --- | --- | --- | --- | --- |
| C1 | clean | clean | `REQUIRE_AGREEMENT` | answer correctly under consistent evidence | no |
| C2 | `DIFFERENT` | clean | `ABSTAIN` | contradiction case; abstain under multimodal conflict | yes |
| C3 | `IRRELEVANT` | clean | `TRUST_VISION` | trust vision answer | yes |
| C4 | clean | irrelevant | `TRUST_TEXT` | trust text answer | no |
| C5 | `IRRELEVANT` | irrelevant | `ABSTAIN` | abstain (neither modality provides reliable evidence) | yes |

Under the nominal refined plan, the target dataset size is `45,000` with equal balancing across three families and five categories. The currently published HF release used in the active workflow contains `44,982` rows (`nbso/carm-vqa-5way`) due to realized post-filtering and moderation outcomes.

| Family | C1 | C2 | C3 | C4 | C5 | Family total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `existence` | 3,000 | 2,995 | 2,995 | 3,000 | 2,997 | 14,987 |
| `count` | 3,000 | 2,998 | 3,000 | 3,000 | 3,000 | 14,998 |
| `attribute_color` | 3,000 | 2,999 | 2,999 | 3,000 | 2,999 | 14,997 |
| Category total | 9,000 | 8,992 | 8,994 | 9,000 | 8,996 | 44,982 |

The realized split allocation for the current prepared release is summarized below.

| Category / Dataset | Train | Val | Test (`test_id`) | Total |
| --- | ---: | ---: | ---: | ---: |
| C1 | 6,300 | 1,350 | 1,350 | 9,000 |
| C2 | 6,293 | 1,347 | 1,352 | 8,992 |
| C3 | 6,295 | 1,348 | 1,351 | 8,994 |
| C4 | 6,300 | 1,350 | 1,350 | 9,000 |
| C5 | 6,296 | 1,348 | 1,352 | 8,996 |
| Full dataset | 31,484 | 6,743 | 6,755 | 44,982 |

Caption-edit workload follows directly from the category design. LLM caption perturbation is required for C2, C3, and C5. In the realized `44,982` release, this corresponds to `26,982` caption edits (`8,992 + 8,994 + 8,996`), with `DIFFERENT:IRRELEVANT = 8,992:17,990` (approximately `1:2.00`). Image-side irrelevance for C4 and C5 is generated through deterministic image swapping rather than blur/occlusion severity edits.

### A7. Illustrative Raw-to-Constructed Examples and Category Manipulation Demonstration

This appendix subsection provides both family-level examples and an explicit category-level demonstration of how each C1-C5 variant is constructed from a clean anchor record. The category protocol is implemented through controlled text-side and image-side manipulations, while preserving deterministic oracle-action assignment.

| Category | Image state | Caption state | Manipulation path | Effective input construction | Oracle action |
| --- | --- | --- | --- | --- | --- |
| C1 | clean | clean | no perturbation | original image + clean caption | `REQUIRE_AGREEMENT` |
| C2 | clean | different | LLM caption rewrite to semantically plausible but conflicting content | original image + `DIFFERENT` perturbed caption | `ABSTAIN` |
| C3 | clean | irrelevant | LLM caption rewrite to unrelated content | original image + `IRRELEVANT` perturbed caption | `TRUST_VISION` |
| C4 | irrelevant | clean | deterministic image swap to an irrelevant donor image | swapped image + clean caption | `TRUST_TEXT` |
| C5 | irrelevant | irrelevant | combined C3 (text-side irrelevance) + C4 (image-side irrelevance) | swapped image + `IRRELEVANT` perturbed caption | `ABSTAIN` |

The following schema-level example illustrates how one clean anchor can be expanded into all five categories under this manipulation policy. For C2, the prepared row stores unimodal support targets at top level rather than hiding text-only truth in metadata; in the current caption-derived export, `text_supported_target` is populated when the contradictory caption supports a concrete answer and otherwise remains null with an explicit provenance flag.

```json
{
  "anchor_example": {
    "example_id": "vqa-100000002::clean",
    "question": "Is the cat wearing a collar?",
    "clean_caption": "A cat sitting next to a wii controller, upside down.",
    "image_state": "clean",
    "caption_state": "clean"
  },
  "category_variants": [
    {
      "protocol_category": "C1",
      "image_state": "clean",
      "caption_state": "clean",
      "text_input_source": "clean_caption",
      "image_source": "original_image",
      "oracle_action": "require_agreement"
    },
    {
      "protocol_category": "C2",
      "image_state": "clean",
      "caption_state": "different",
      "text_input_source": "perturbed_caption_different",
      "image_source": "original_image",
      "oracle_action": "abstain",
      "vision_supported_target": "yes",
      "text_supported_target": "no"
    },
    {
      "protocol_category": "C3",
      "image_state": "clean",
      "caption_state": "irrelevant",
      "text_input_source": "perturbed_caption_irrelevant",
      "image_source": "original_image",
      "oracle_action": "trust_vision"
    },
    {
      "protocol_category": "C4",
      "image_state": "irrelevant",
      "caption_state": "clean",
      "text_input_source": "clean_caption",
      "image_source": "swapped_irrelevant_image",
      "oracle_action": "trust_text"
    },
    {
      "protocol_category": "C5",
      "image_state": "irrelevant",
      "caption_state": "irrelevant",
      "text_input_source": "perturbed_caption_irrelevant",
      "image_source": "swapped_irrelevant_image",
      "oracle_action": "abstain"
    }
  ]
}
```

The family-specific examples below illustrate representative mappings from raw VQAv2/COCO annotations to normalized clean base records for the three active families; each base record is subsequently expanded into the C1-C5 variants using the manipulation rules above.

```json
{
  "vqa_question_raw": {
    "question_id": 100000002,
    "image_id": 100000,
    "question": "Is the cat wearing a collar?"
  },
  "vqa_annotation_raw": {
    "multiple_choice_answer": "yes"
  },
  "coco_caption_raw": {
    "caption": "A cat sitting next to  a wii controller, upside down."
  },
  "constructed_base_example": {
    "example_id": "vqa-100000002::clean",
    "family": "existence",
    "gold_answer": "yes"
  }
}
```

![Existence example image](data/interim/example_images/existence_COCO_val2014_000000100000.jpg)

```json
{
  "vqa_question_raw": {
    "question_id": 100022005,
    "image_id": 100022,
    "question": "How many baskets are there?"
  },
  "vqa_annotation_raw": {
    "multiple_choice_answer": "2"
  },
  "coco_caption_raw": {
    "caption": "two pink bowls with food in it, rice and tomatoes "
  },
  "constructed_base_example": {
    "example_id": "vqa-100022005::clean",
    "family": "count",
    "gold_answer": "2"
  }
}
```

![Count example image](data/interim/example_images/count_COCO_train2014_000000100022.jpg)

```json
{
  "vqa_question_raw": {
    "question_id": 100012011,
    "image_id": 100012,
    "question": "What color is the shirt of the goalkeeper?"
  },
  "vqa_annotation_raw": {
    "multiple_choice_answer": "white"
  },
  "coco_caption_raw": {
    "caption": "Two men in field catching a white frisbee."
  },
  "constructed_base_example": {
    "example_id": "vqa-100012011::clean",
    "family": "attribute_color",
    "gold_answer": "white"
  }
}
```

![Attribute-color example image](data/interim/example_images/attribute_color_COCO_train2014_000000100012.jpg)

## Appendix B. Evaluation Details

### B1. Splits, Baselines, and Per-Example Records

The evaluation protocol uses one constructed dataset with deterministic `train`, `val`, and `test` (`test_id`) partitions. The `train` split is reserved for later CARM model training, while baseline development is performed on `val` and final reporting is performed on `test` (`test_id`) only. Development evaluation on `val` is used for threshold selection and sanity checks. Final evaluation on `test` (`test_id`) is run with all choices locked, with no additional tuning.

The active baseline comparison set contains four methods: Direct (`backbone_direct`), Agreement check (`agreement_check`), Confidence threshold (`confidence_threshold`), and Probe heuristic (`probe_heuristic`). The Direct baseline returns the backbone response from full multimodal input (image, caption, and question) without abstention. The Agreement-check baseline generates vision-only and text-only probe answers and abstains when those answers disagree. The Confidence-threshold baseline computes inverse normalized entropy from the multimodal answer distribution and abstains below a tuned threshold. The Probe-heuristic baseline runs vision-only and text-only probes, routes to the lower-entropy probe answer, and abstains only when both probes are sufficiently uncertain.

For each example, every baseline emits a standardized core record containing input and gold fields (`example_id`, `base_id`, `image_path`, `text_input`, `question`, `gold_answer`, `split`, `family`, `oracle_action`, and `protocol_category`), prediction fields (`final_answer`, `abstained`, `confidence`), and derived evaluation fields (`correct`, `task_success`). This per-example table is treated as the single source of truth for all aggregate metrics and plots. When evaluating CARM, optional extra diagnostic fields may be appended, but they are not required by the baseline workflow.

### B2. Answering and Task-Success Metrics

Because the four baselines do not emit routing actions, the main baseline table does not report action-classification metrics. Instead, it focuses on answering quality, abstention behavior, and outcome-aligned task success. When CARM is evaluated, action-aware diagnostics can still be computed from the optional appended predictor fields.

Coverage is defined as:

$$
\mathrm{Coverage}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\neg \mathrm{abstained}_i].
$$

Abstain rate is reported as:

$$
1-\mathrm{Coverage}.
$$

Accuracy-on-answered is:

$$
\mathrm{AccAnswered}=
\frac{\sum_{i=1}^{N}\mathbf{1}[\neg \mathrm{abstained}_i \land \mathrm{correct}_i]}
{\sum_{i=1}^{N}\mathbf{1}[\neg \mathrm{abstained}_i]}.
$$

Raw answer accuracy alone is insufficient under C2/C5 behavior, so the primary project metric is task success. For flat baseline records, task success is defined as:

$$
\mathrm{task\_success}_i=
\begin{cases}
1, & \text{if } \mathrm{protocol\_category}_i\in\{\mathrm{C2},\mathrm{C5}\}\ \land\ \mathrm{abstained}_i, \\
1, & \text{if } \mathrm{protocol\_category}_i\in\{\mathrm{C1},\mathrm{C3},\mathrm{C4}\}\ \land\ \neg \mathrm{abstained}_i \land \mathrm{correct}_i, \\
1, & \text{if } \mathrm{protocol\_category}_i=\varnothing \land \mathrm{oracle\_action}_i=\mathrm{require\_agreement} \land (\mathrm{abstained}_i \lor \mathrm{correct}_i), \\
1, & \text{if } \mathrm{protocol\_category}_i=\varnothing \land \mathrm{oracle\_action}_i\in\{\mathrm{trust\_vision},\mathrm{trust\_text}\} \land \neg \mathrm{abstained}_i \land \mathrm{correct}_i, \\
0, & \text{otherwise}.
\end{cases}
$$

Task success rate is then:

$$
\mathrm{TaskSuccess}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{task\_success}_i].
$$

### B3. Selective Prediction and Confidence Standardization

Selective prediction behavior is evaluated by sweeping thresholds over a scalar confidence score. At threshold $\tau$, examples below threshold are forcibly abstained and examples above threshold are retained; this yields coverage$(\tau)$ and risk$(\tau)$. The primary risk curve is defined on `task_success` rather than raw correctness. Lower risk at fixed coverage indicates better selective behavior. We summarize the curve using `Risk@80% coverage` (or the nearest achievable point) and `AURC` (area under the risk-coverage curve, lower is better).

The primary confidence definition is inverse normalized entropy:

$$
\mathrm{confidence}=1-\mathrm{normalized\_entropy}(p),
$$

where $p$ is the answer distribution used by the baseline for its decision path (vision-only, text-only, or multimodal). For the generation-based backbone, this distribution is derived from the parsed answer path and the model's token-level generation scores.

### B4. Category-Level Interpretation and Locked Workflow

In addition to aggregate metrics, we report per-category (`C1`-`C5`) breakdowns for Coverage, AccAnswered, Accuracy, and TaskSuccess. This category view is required for interpretation because each category captures a distinct arbitration behavior: C2 and C5 should exhibit abstention-driven task success, C3 should reward vision-grounded correctness, C4 should reward text-grounded correctness, and C1 should reward correct answering under consistent evidence. For C2, we additionally report vision-only accuracy, text-only accuracy, and multimodal abstention rate as secondary diagnostics. The current caption-derived export makes these diagnostics partially reportable rather than all-or-nothing: on the realized `44,982`-row release, `8,931 / 8,992` C2 rows (`99.32%`) receive a caption-derived `text_supported_target`, while `61` remain explicitly missing.

The baseline workflow proceeds in four locked stages. First, all baselines are run on `val` to produce per-example outputs and aggregate diagnostics. Second, only thresholded methods (`confidence_threshold` and `probe_heuristic`) are tuned on `val` by maximizing `TaskSuccess`; ties are broken by higher Coverage, then by the least aggressive abstention setting (`lowest confidence_threshold`, `highest probe_both_uncertain_threshold`). This step writes a frozen `tuned_thresholds.json` artifact. Third, all settings are frozen and a single final run is executed on `test` (`test_id`) with `--tuned-thresholds-json` and no additional tuning. Fourth, reporting includes a main table (Accuracy, Coverage, AccAnswered, TaskSuccess, and Risk@target/AURC), risk-coverage plot(s), the C1-C5 breakdown tables, and the C2 diagnostics table. The C2 table now prints each metric with its evaluable denominator, so the partial text-target coverage of the caption-derived export is visible directly in the report rather than being hidden behind blanket `n/a` values.
