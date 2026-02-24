from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from carm.data.schema import ConflictExample, CorruptModality, Family, Split


class IntegrityError(ValueError):
    """Raised when split integrity checks fail."""


def compute_manifest_hash(example_ids: list[str]) -> str:
    payload = "\n".join(sorted(example_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_by_split(examples: list[ConflictExample]) -> dict[str, str]:
    split_ids: dict[Split, list[str]] = defaultdict(list)
    for ex in examples:
        split_ids[ex.split].append(ex.example_id)
    return {
        split.value: compute_manifest_hash(ids)
        for split, ids in sorted(split_ids.items(), key=lambda kv: kv[0].value)
    }


def _count_by_split(examples: list[ConflictExample]) -> dict[str, int]:
    counts = Counter(ex.split.value for ex in examples)
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def validate_split_integrity(
    examples: list[ConflictExample],
    heldout_family: Family | None = None,
    heldout_severity: int = 3,
    enforce_template_disjointness: bool = False,
) -> dict[str, Any]:
    seen_ids: set[str] = set()
    split_images: dict[Split, set[str]] = defaultdict(set)
    split_templates: dict[Split, set[str]] = defaultdict(set)

    for ex in examples:
        if ex.example_id in seen_ids:
            raise IntegrityError(f"Duplicate example_id: {ex.example_id}")
        seen_ids.add(ex.example_id)

        if ex.corrupt_modality == CorruptModality.NONE and ex.severity != 0:
            raise IntegrityError(f"Severity must be 0 for non-corrupted examples: {ex.example_id}")

        source_key = ex.source_image_id or ex.image_path.split("|")[0]
        split_images[ex.split].add(source_key)

        if ex.template_id:
            split_templates[ex.split].add(ex.template_id)

        if ex.split in {Split.TRAIN, Split.VAL, Split.TEST_ID}:
            if ex.heldout_family_flag:
                raise IntegrityError(
                    f"heldout_family_flag cannot be true in ID splits: {ex.example_id}"
                )
            if ex.heldout_severity_flag:
                raise IntegrityError(
                    f"heldout_severity_flag cannot be true in ID splits: {ex.example_id}"
                )

        if heldout_family and ex.split == Split.TEST_OOD_FAMILY:
            if ex.family != heldout_family:
                raise IntegrityError(
                    f"OOD-family split contains non-heldout family example: {ex.example_id}"
                )
            if not ex.heldout_family_flag:
                raise IntegrityError(
                    f"OOD-family example missing heldout_family_flag: {ex.example_id}"
                )

        if ex.split == Split.TEST_OOD_SEVERITY:
            if ex.corrupt_modality == CorruptModality.NONE or ex.severity < heldout_severity:
                raise IntegrityError(
                    f"OOD-severity split has invalid example (needs corrupted severity >= {heldout_severity}): {ex.example_id}"
                )
            if not ex.heldout_severity_flag:
                raise IntegrityError(
                    f"OOD-severity example missing heldout_severity_flag: {ex.example_id}"
                )

    id_splits = [Split.TRAIN, Split.VAL, Split.TEST_ID]
    for i, left in enumerate(id_splits):
        for right in id_splits[i + 1 :]:
            overlap = split_images[left].intersection(split_images[right])
            if overlap:
                sample = sorted(overlap)[0]
                raise IntegrityError(
                    f"Image-source leakage between {left.value} and {right.value}: {sample}"
                )

    if enforce_template_disjointness:
        for i, left in enumerate(id_splits):
            for right in id_splits[i + 1 :]:
                overlap = split_templates[left].intersection(split_templates[right])
                if overlap:
                    sample = sorted(overlap)[0]
                    raise IntegrityError(
                        f"Template leakage between {left.value} and {right.value}: {sample}"
                    )

    return {
        "hashes": _hash_by_split(examples),
        "counts": _count_by_split(examples),
        "total_examples": len(examples),
    }
