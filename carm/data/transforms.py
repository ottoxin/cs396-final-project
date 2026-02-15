from __future__ import annotations

import random
import re
from dataclasses import replace

from carm.data.schema import ConflictExample, ConflictType, CorruptedModality


ATTRIBUTE_FLIPS = {
    "red": "blue",
    "blue": "red",
    "black": "white",
    "white": "black",
    "small": "large",
    "large": "small",
}

RELATION_FLIPS = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
}


def _derived_id(base_id: str, suffix: str) -> str:
    return f"{base_id}::{suffix}"


def caption_swap(
    example: ConflictExample,
    donor_text: str,
    same_topic: bool = True,
    seed: int | None = None,
) -> ConflictExample:
    rng = random.Random(seed)
    family = "caption_swap_same_topic" if same_topic else "caption_swap"
    swapped = replace(
        example,
        example_id=_derived_id(example.example_id, "swap"),
        text_input=donor_text,
        conflict_type=ConflictType.OBJECT,
        corrupted_modality=CorruptedModality.TEXT,
        corruption_family=family,
        severity=1,
    )
    swapped.metadata = {**example.metadata, "swap_seed": rng.randint(0, 10_000)}
    return swapped


def typed_text_edit(example: ConflictExample, conflict_type: ConflictType) -> ConflictExample:
    text = example.text_input
    family = "text_edit"
    if conflict_type == ConflictType.ATTRIBUTE:
        for src, dst in ATTRIBUTE_FLIPS.items():
            pattern = rf"\b{re.escape(src)}\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                text = re.sub(pattern, dst, text, count=1, flags=re.IGNORECASE)
                break
        family = "text_edit_attribute"
    elif conflict_type == ConflictType.RELATION:
        for src, dst in RELATION_FLIPS.items():
            pattern = rf"\b{re.escape(src)}\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                text = re.sub(pattern, dst, text, count=1, flags=re.IGNORECASE)
                break
        family = "text_edit_relation"
    elif conflict_type == ConflictType.COUNT:
        text = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1), text, count=1)
        family = "text_edit_count"

    edited = replace(
        example,
        example_id=_derived_id(example.example_id, f"edit_{conflict_type.value}"),
        text_input=text,
        conflict_type=conflict_type,
        corrupted_modality=CorruptedModality.TEXT,
        corruption_family=family,
        severity=1,
    )
    edited.metadata = {**example.metadata, "edit_type": conflict_type.value}
    return edited


def vision_corruption(example: ConflictExample, mode: str, severity: int) -> ConflictExample:
    if mode not in {"blur", "occlusion", "distractor"}:
        raise ValueError(f"Unknown vision corruption mode: {mode}")
    image_path = f"{example.image_path}|{mode}|s{severity}"
    corrupted = replace(
        example,
        example_id=_derived_id(example.example_id, f"vision_{mode}_s{severity}"),
        image_path=image_path,
        conflict_type=ConflictType.OBJECT,
        corrupted_modality=CorruptedModality.VISION,
        corruption_family=f"vision_{mode}",
        severity=severity,
    )
    corrupted.metadata = {**example.metadata, "vision_corruption": mode}
    return corrupted
