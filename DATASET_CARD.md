# Dataset Card: CARM Conflict Suite v1 (Class-Medium Real Vision Pilot)

## Summary

This is the class-medium prebuilt release of Conflict Suite v1 for multimodal conflict arbitration in VQA.
It is derived from VQAv2 questions/answers and COCO captions, then expanded with deterministic conflict operators:
`swap_easy`, `swap_hard`, `text_edit`, and `vision_corrupt`.

Release snapshot (2026-02-28):
- `pilot_3k_class_medium_real_vision.jsonl` (`21,000` rows)
- `vision_corrupt/class_medium/pilot_3k/` (`9,000` occluded images)
- manifests for deterministic reconstruction and integrity checks

## Intended Use

Use for research/education evaluation of:
- conflict-aware action selection (`trust_vision`, `trust_text`, `require_agreement`, `abstain`)
- OOD-family and OOD-severity robustness
- calibration/selective prediction under disagreement

Not for high-stakes decision deployment.

## Splits and Operators

Pilot split counts (`pilot_3k_class_medium.manifest.json`):
- `train`: 8,304
- `val`: 1,866
- `test_id`: 1,830
- `test_ood_family`: 7,000
- `test_ood_severity`: 2,000

Pilot operator counts:
- `clean`: 3,000
- `swap_easy`: 3,000
- `swap_hard`: 3,000
- `text_edit`: 3,000
- `vision_corrupt`: 9,000

## Data Fields

Core JSONL fields include:
- IDs: `example_id`, `base_id`, `variant_id`, `source_image_id`
- Inputs: `image_path`, `text_input`, `question`
- Labels/metadata: `gold_answer`, `family`, `operator`, `corrupt_modality`, `severity`, `oracle_action`
- Split/OOD flags: `split`, `heldout_family_flag`, `heldout_severity_flag`, `hard_swap_flag`

## Creation Process

1. Start from official VQAv2 + COCO captions.
2. Keep families `existence`, `count`, `attribute_color`.
3. Apply answer normalization and caption-consistency filtering.
4. Generate deterministic conflict variants.
5. Assign source-image-disjoint splits with OOD overrides.
6. Materialize deterministic pixel-level occlusions for pilot vision rows.

## Reproducibility

From `pilot_3k_class_medium_real_vision.manifest.json`:
- output JSONL SHA256: `42e677db5c99c657ac44946749c0c800c06df4ec7fe9d8fc919ac94c7e92b990`
- image-dir fingerprint SHA256: `fc835c8f7e7aa24f6b22007f8596f3f24a5e28f58841b5aecbf2282d1b873123`
- image file count: `9,000`

## Original-Source Note

- This release is a derived artifact package, not a full mirror of VQAv2/COCO.
- Rows with operators `clean`, `swap_easy`, `swap_hard`, and `text_edit` reference original COCO image paths under `data/raw/coco/...`.
- For full inference on all pilot rows, download official COCO train/val images locally.

## Licensing

This dataset is derived from VQAv2 and COCO. Users must comply with upstream licenses/terms when using, sharing, or redistributing derived artifacts.

