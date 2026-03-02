# Conflict Suite v1 Data Construction Note

## Dataset Construction

Conflict Suite v1 is constructed from official VQAv2 question-answer annotations and COCO captions, with each example anchored to a single COCO image. The construction objective is to create a high-validity benchmark for multimodal conflict handling under controlled perturbations while preserving deterministic supervision and reproducible splits. In the current protocol, we retain three question families that support reliable normalization and agreement checks: `existence`, `count`, and `attribute_color`. Detailed stage-by-stage specifications are provided in Appendix A1-A5.

For each VQAv2 question, we infer family membership from question prefixes and then normalize the gold answer under family-specific rules. Existence answers are normalized to `yes` or `no`, count answers are normalized to integer strings, and color answers are restricted to a fixed closed vocabulary. Any question that does not map to one of the selected families, or any answer that cannot be normalized under the active family rule, is excluded before variant generation (Appendix A1).

After normalization, we enforce caption consistency. Given all COCO captions for the image tied to a question, we retain the first caption that is compatible with the normalized answer under family-specific support rules. If no caption satisfies the rule, the example is dropped with a consistency-filter failure reason. This step produces clean base examples where the retained text evidence is compatible with the answer used for supervision (Appendix A2).

From each retained base example, we generate Conflict Suite v1 variants with deterministic metadata and oracle actions. The emitted set contains one clean variant, two caption-swap variants (`swap_easy` and `swap_hard`), one family-aligned text-edit variant, and three vision-corruption variants with severities 1, 2, and 3. The hard-swap operator uses constrained donor selection and falls back to easy-swap donor text when no valid hard donor is available, while preserving the `SWAP_HARD` operator tag and exposing fallback status through `hard_swap_flag` (Appendix A3). Oracle-action mapping rules are summarized in Appendix A4.

Split assignment is performed at source-image level to prevent leakage across variants derived from the same image. Base splits are assigned with deterministic seeded shuffling and default 70/15/15 source ratios for train/val/test-ID, then overwritten by protocol-level OOD rules for held-out family and held-out severity. The generated manifest records counts, hashes, configuration values, and integrity checks to support exact regeneration (Appendix A5).

In the current official-data build with consistency filtering enabled, the base-construction stage processed 658,111 candidate VQAv2 question-answer records and retained 184,590 clean base examples (28.0%), while filtering out 473,521 records (72.0%). The filtered set consists of 308,623 records removed by family gating, 22,060 removed by answer-normalization failure, and 142,838 removed by caption-consistency failure. The retained family composition is 150,582 existence examples, 15,207 count examples, and 18,801 attribute-color examples.

### CARM Supervision Refinement

The project objective is to train the CARM module to select among four entropy-conditioned actions: high-entropy vision and high-entropy text (`ABSTAIN`), low-entropy vision and high-entropy text (`TRUST_VISION`), high-entropy vision and low-entropy text (`TRUST_TEXT`), and low-entropy vision and low-entropy text (require agreement and return an answer only when both modalities agree; otherwise abstain).

Under the current text-perturbation strategy, captions are often modified so that they imply a different answer while remaining equally plausible as evidence for the question. Assigning these cases to `TRUST_VISION` can create a misleading training signal, because both modalities may appear low entropy yet disagree semantically.

To provide finer-grained supervision, we propose two text-edit categories. `IRRELEVANT` edits remove or alter answer-bearing caption content so that text no longer supports answering the question; these cases should map to `TRUST_VISION`. `DIFFERENT` edits preserve answerability but change the implied answer (for example, replacing color or count tokens); these cases should follow the same oracle policy as clean low-entropy/low-entropy examples, namely agreement checking with abstention on disagreement.

A practical data-construction recipe is a balanced three-way mix: one-third `clean`, one-third `IRRELEVANT`, and one-third `DIFFERENT`, generated in API batches (for example, with GPT-5-nano). In this setup, `clean` and `DIFFERENT` share the same oracle policy, while `IRRELEVANT` maps to `TRUST_VISION`. The current working estimate for text disruption cost is approximately $0.075 per 1,000 samples under this 1/3-1/3-1/3 split.

Image perturbation remains an open labeling problem. Blur or occlusion does not guarantee that visual evidence becomes unanswerable, so oracle actions are not always known a priori. One option is to use a vision-capable model to determine whether the perturbed image still answers the question and then assign oracle labels accordingly. Another option is COCO mask-guided occlusion: extract question-relevant objects, apply targeted versus non-targeted occlusions, and retain cases where answerability is controlled by construction. This approach is likely most useful for counting and less direct for attribute-color questions.

The following raw-to-constructed examples illustrate the three active families and show how source annotations map into clean base records.

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

Oracle action labels are deterministic functions of corruption modality. Clean examples (`none`) map to `REQUIRE_AGREEMENT`, text-corrupted examples map to `TRUST_VISION`, and vision-corrupted examples map to `TRUST_TEXT`. Optional both-corrupted examples, when enabled with ambiguity, map to `ABSTAIN`. This mapping is applied uniformly across all generated records and is independent of model predictions.

### A5. Split Assignment, OOD Overrides, and Manifests

Split construction begins with source-image-disjoint base assignment. Unique source image identifiers are shuffled with a fixed seed and partitioned with default 70/15/15 ratios into train, validation, and test-ID sources. Every variant derived from a source image inherits this base split before OOD overrides are applied.

OOD assignment then follows fixed priority: records in the held-out family are assigned to `TEST_OOD_FAMILY`; otherwise, corrupted records at or above held-out severity are assigned to `TEST_OOD_SEVERITY`; and optional hard-swap OOD assignment is applied when enabled and when `hard_swap_flag` is true. Final artifacts include split-wise counts, split hashes, configuration echoes, and integrity-validation outputs, enabling deterministic regeneration and leakage checks.
