from __future__ import annotations

import hashlib
from collections import defaultdict

from carm.data.schema import ConflictExample, CorruptedModality, Split


class IntegrityError(ValueError):
    """Raised when split integrity checks fail."""


def compute_manifest_hash(example_ids: list[str]) -> str:
    payload = "\n".join(sorted(example_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_split_integrity(examples: list[ConflictExample]) -> dict[str, str]:
    seen: set[str] = set()
    split_ids: dict[Split, list[str]] = defaultdict(list)
    split_images: dict[Split, set[str]] = defaultdict(set)
    split_templates: dict[Split, set[str]] = defaultdict(set)

    for ex in examples:
        if ex.example_id in seen:
            raise IntegrityError(f"Duplicate example_id: {ex.example_id}")
        seen.add(ex.example_id)
        split_ids[ex.split].append(ex.example_id)

        image_key = ex.source_image_id or ex.image_path
        if image_key in split_images[ex.split]:
            pass
        split_images[ex.split].add(image_key)

        if ex.template_id:
            split_templates[ex.split].add(ex.template_id)

        if ex.corrupted_modality == CorruptedModality.NONE and ex.severity != 0:
            raise IntegrityError(
                f"Severity must be 0 for non-corrupted examples: {ex.example_id}"
            )

    # Image-source disjointness across splits.
    splits = [Split.TRAIN, Split.VAL, Split.TEST]
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = split_images[left].intersection(split_images[right])
            if overlap:
                sample = sorted(list(overlap))[0]
                raise IntegrityError(
                    f"Image-source leakage between {left.value} and {right.value}: {sample}"
                )

    # Conflict-template disjointness across splits where template_id is provided.
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = split_templates[left].intersection(split_templates[right])
            if overlap:
                sample = sorted(list(overlap))[0]
                raise IntegrityError(
                    f"Template leakage between {left.value} and {right.value}: {sample}"
                )

    return {
        split.value: compute_manifest_hash(split_ids.get(split, []))
        for split in splits
    }
