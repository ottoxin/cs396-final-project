from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


FAMILY_MAP = {
    "existence": "existence",
    "count": "count",
    "attribute_color": "attribute_color",
}

ACTION_ALIAS = {
    "trust_image": "trust_vision",
    "trust_vision": "trust_vision",
    "trust_text": "trust_text",
    "require_agreement": "require_agreement",
    "abstain": "abstain",
}

TUPLE_TO_CATEGORY = {
    ("clean", "clean", "require_agreement"): "C1",
    ("clean", "different", "require_agreement"): "C2",
    ("clean", "irrelevant", "trust_vision"): "C3",
    ("irrelevant", "clean", "trust_text"): "C4",
    ("irrelevant", "irrelevant", "abstain"): "C5",
}

CATEGORY_TO_SCHEMA = {
    "C1": ("clean", "none", 0, "require_agreement"),
    "C2": ("text_edit", "text", 1, "require_agreement"),
    "C3": ("text_edit", "text", 1, "trust_vision"),
    "C4": ("vision_corrupt", "vision", 1, "trust_text"),
    "C5": ("both", "both", 1, "abstain"),
}


def normalize_oracle_action(action: str) -> str:
    normalized = ACTION_ALIAS.get(action.strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported oracle_action: {action}")
    return normalized


def derive_protocol_category(image_state: str, caption_state: str, oracle_action: str) -> str:
    key = (
        image_state.strip().lower(),
        caption_state.strip().lower(),
        normalize_oracle_action(oracle_action),
    )
    out = TUPLE_TO_CATEGORY.get(key)
    if out is None:
        raise ValueError(f"Unsupported category tuple: {key}")
    return out


def answer_type_for_family(family: str) -> str:
    normalized = family.strip().lower()
    if normalized == "existence":
        return "boolean"
    if normalized == "count":
        return "integer"
    if normalized == "attribute_color":
        return "color"
    raise ValueError(f"Unsupported family: {family}")


def schema_fields_for_category(category: str) -> tuple[str, str, int, str]:
    out = CATEGORY_TO_SCHEMA.get(category)
    if out is None:
        raise ValueError(f"Unsupported protocol category: {category}")
    return out


def choose_text_input(caption_state: str, clean_caption: str, perturbed_caption: str | None) -> str:
    if caption_state.strip().lower() == "clean":
        return str(clean_caption)
    if perturbed_caption is None or not str(perturbed_caption).strip():
        raise ValueError("Missing perturbed caption for non-clean caption_state.")
    return str(perturbed_caption)


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def normalized(self) -> tuple[float, float, float]:
        total = self.train + self.val + self.test
        if total <= 0.0:
            raise ValueError("Split ratios must sum to a positive value.")
        return self.train / total, self.val / total, self.test / total


def assign_splits_by_base(
    rows: list[dict],
    *,
    seed: int,
    ratios: SplitRatios,
) -> dict[str, str]:
    train_r, val_r, _ = ratios.normalized()
    bases: dict[str, dict[str, object]] = {}
    for row in rows:
        base_id = str(row["base_id"])
        family = str(row["family"])
        category = str(row["protocol_category"])
        entry = bases.setdefault(base_id, {"family": family, "categories": set()})
        categories = entry["categories"]
        assert isinstance(categories, set)
        categories.add(category)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for base_id, entry in bases.items():
        family = str(entry["family"])
        categories = entry["categories"]
        assert isinstance(categories, set)
        signature = "+".join(sorted(categories))
        strata[(family, signature)].append(base_id)

    assignment: dict[str, str] = {}
    for key, base_ids in sorted(strata.items(), key=lambda kv: kv[0]):
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        ordered = sorted(base_ids)
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        n_test = n - n_train - n_val
        if n > 0 and n_test <= 0:
            n_test = 1
            if n_train > n_val and n_train > 0:
                n_train -= 1
            elif n_val > 0:
                n_val -= 1
        for idx, base_id in enumerate(ordered):
            if idx < n_train:
                assignment[base_id] = "train"
            elif idx < n_train + n_val:
                assignment[base_id] = "val"
            else:
                assignment[base_id] = "test_id"
    return assignment
