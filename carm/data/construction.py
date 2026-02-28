from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any

from carm.data.integrity import validate_split_integrity
from carm.data.labeling import derive_oracle_action
from carm.data.schema import (
    AnswerType,
    ConflictExample,
    CorruptModality,
    Family,
    Operator,
    Split,
)
from carm.data.transforms import STOPWORDS, caption_swap, text_edit, vision_corrupt


def infer_source_image_id(image_path: str) -> str:
    return image_path.split("|")[0]


def infer_template_id(question: str) -> str:
    normalized = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _noun_like_tokens(text: str, cache: dict[str, set[str]]) -> set[str]:
    cached = cache.get(text)
    if cached is not None:
        return cached
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = {t for t in toks if t not in STOPWORDS and len(t) > 2}
    cache[text] = out
    return out


def _jaccard_tokens(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a.intersection(b)) / max(1, len(a.union(b)))


def _set_clean_defaults(example: ConflictExample) -> ConflictExample:
    out = replace(
        example,
        example_id=f"{example.base_id}::clean",
        variant_id="clean",
        operator=Operator.CLEAN,
        corrupt_modality=CorruptModality.NONE,
        severity=0,
        hard_swap_flag=False,
        heldout_family_flag=False,
        heldout_severity_flag=False,
        split=Split.TRAIN,
        source_image_id=example.source_image_id or infer_source_image_id(example.image_path),
        template_id=example.template_id or infer_template_id(example.question),
    )
    out.oracle_action = derive_oracle_action(out.corrupt_modality)
    return out


def _source_split(
    examples: list[ConflictExample],
    seed: int,
    ratios: dict[str, float],
) -> dict[str, Split]:
    rng = random.Random(seed)
    source_ids = sorted({ex.source_image_id or infer_source_image_id(ex.image_path) for ex in examples})
    rng.shuffle(source_ids)

    n = len(source_ids)
    n_train = int(n * float(ratios.get("train", 0.7)))
    n_val = int(n * float(ratios.get("val", 0.15)))

    train_sources = set(source_ids[:n_train])
    val_sources = set(source_ids[n_train : n_train + n_val])

    split_map: dict[str, Split] = {}
    for source in source_ids:
        if source in train_sources:
            split_map[source] = Split.TRAIN
        elif source in val_sources:
            split_map[source] = Split.VAL
        else:
            split_map[source] = Split.TEST_ID
    return split_map


def _assign_splits(
    examples: list[ConflictExample],
    seed: int,
    ratios: dict[str, float],
    held_out_family: Family,
    held_out_severity: int,
    enable_ood_hard_swap: bool,
) -> list[ConflictExample]:
    source_map = _source_split(examples, seed=seed, ratios=ratios)

    out: list[ConflictExample] = []
    for ex in examples:
        source = ex.source_image_id or infer_source_image_id(ex.image_path)
        split = source_map[source]

        heldout_family_flag = False
        heldout_severity_flag = False

        if ex.family == held_out_family:
            split = Split.TEST_OOD_FAMILY
            heldout_family_flag = True
        elif ex.corrupt_modality != CorruptModality.NONE and ex.severity >= held_out_severity:
            split = Split.TEST_OOD_SEVERITY
            heldout_severity_flag = True
        elif enable_ood_hard_swap and ex.operator == Operator.SWAP_HARD and ex.hard_swap_flag:
            split = Split.TEST_OOD_HARD_SWAP

        out.append(
            replace(
                ex,
                split=split,
                heldout_family_flag=heldout_family_flag,
                heldout_severity_flag=heldout_severity_flag,
            )
        )
    return out


def _hard_swap_candidate(
    base: ConflictExample,
    donors: list[ConflictExample],
    jaccard_min: float,
    jaccard_max: float,
    rng: random.Random,
    token_cache: dict[str, set[str]],
) -> ConflictExample | None:
    base_tokens = _noun_like_tokens(base.text_input, token_cache)
    candidates: list[ConflictExample] = []
    for donor in donors:
        if donor.base_id == base.base_id:
            continue
        if (donor.source_image_id or "") == (base.source_image_id or ""):
            continue
        donor_tokens = _noun_like_tokens(donor.text_input, token_cache)
        score = _jaccard_tokens(base_tokens, donor_tokens)
        if score < jaccard_min or score > jaccard_max:
            continue
        candidates.append(donor)

    if not candidates:
        return None
    return rng.choice(candidates)


def build_conflict_suite(
    base_examples: list[ConflictExample],
    *,
    seed: int = 7,
    held_out_family: Family = Family.ATTRIBUTE_COLOR,
    held_out_severity: int = 3,
    split_ratios: dict[str, float] | None = None,
    vision_corruption_type: str = "occlusion",
    vision_severities: list[int] | None = None,
    color_vocab: list[str] | None = None,
    hard_swap_jaccard_min: float = 0.2,
    hard_swap_jaccard_max: float = 0.7,
    include_both_variants: bool = False,
    enable_ood_hard_swap: bool = False,
    enforce_template_disjointness: bool = False,
) -> tuple[list[ConflictExample], dict[str, Any]]:
    if split_ratios is None:
        split_ratios = {"train": 0.7, "val": 0.15, "test_id": 0.15}
    if vision_severities is None:
        vision_severities = [1, 2, 3]
    if color_vocab is None:
        color_vocab = ["red", "blue", "green", "yellow", "black", "white", "brown", "gray"]

    rng = random.Random(seed)

    normalized_base: list[ConflictExample] = []
    for ex in base_examples:
        if ex.operator != Operator.CLEAN:
            clean_ex = replace(ex, operator=Operator.CLEAN, corrupt_modality=CorruptModality.NONE, severity=0)
        else:
            clean_ex = ex
        clean_ex = _set_clean_defaults(clean_ex)
        normalized_base.append(clean_ex)

    # Pre-index donors by (family, answer_type) to avoid repeated full scans in hard swap.
    donors_by_bucket: dict[tuple[Family, AnswerType], list[ConflictExample]] = defaultdict(list)
    for ex in normalized_base:
        donors_by_bucket[(ex.family, ex.answer_type)].append(ex)
    token_cache: dict[str, set[str]] = {}

    generated: list[ConflictExample] = []
    for base in normalized_base:
        generated.append(base)

        easy_pool = [d for d in normalized_base if d.base_id != base.base_id]
        if easy_pool:
            easy_donor = rng.choice(easy_pool)
            generated.append(
                caption_swap(
                    base,
                    donor_text=easy_donor.text_input,
                    operator=Operator.SWAP_EASY,
                    hard_swap_flag=False,
                    seed=seed,
                )
            )

            hard_donor = _hard_swap_candidate(
                base,
                donors=donors_by_bucket[(base.family, base.answer_type)],
                jaccard_min=hard_swap_jaccard_min,
                jaccard_max=hard_swap_jaccard_max,
                rng=rng,
                token_cache=token_cache,
            )
            if hard_donor is None:
                generated.append(
                    caption_swap(
                        base,
                        donor_text=easy_donor.text_input,
                        operator=Operator.SWAP_HARD,
                        hard_swap_flag=False,
                        seed=seed,
                    )
                )
            else:
                generated.append(
                    caption_swap(
                        base,
                        donor_text=hard_donor.text_input,
                        operator=Operator.SWAP_HARD,
                        hard_swap_flag=True,
                        seed=seed,
                    )
                )

        generated.append(text_edit(base, color_vocab=color_vocab, seed=seed))

        for severity in sorted(set(int(s) for s in vision_severities)):
            generated.append(
                vision_corrupt(
                    base,
                    corruption_type=vision_corruption_type,
                    severity=severity,
                )
            )

        if include_both_variants:
            both = replace(
                text_edit(base, color_vocab=color_vocab, seed=seed),
                example_id=f"{base.base_id}::both",
                variant_id="both",
                operator=Operator.BOTH,
                corrupt_modality=CorruptModality.BOTH,
                severity=max(vision_severities),
            )
            both.oracle_action = derive_oracle_action(CorruptModality.BOTH, is_ambiguous=True)
            both.metadata = {
                **both.metadata,
                "both_variant": True,
            }
            generated.append(both)

    assigned = _assign_splits(
        generated,
        seed=seed,
        ratios=split_ratios,
        held_out_family=held_out_family,
        held_out_severity=held_out_severity,
        enable_ood_hard_swap=enable_ood_hard_swap,
    )

    assigned = sorted(assigned, key=lambda ex: ex.example_id)
    manifest = validate_split_integrity(
        assigned,
        heldout_family=held_out_family,
        heldout_severity=held_out_severity,
        enforce_template_disjointness=enforce_template_disjointness,
    )
    manifest["config"] = {
        "seed": seed,
        "held_out_family": held_out_family.value,
        "held_out_severity": held_out_severity,
        "split_ratios": split_ratios,
        "vision_corruption_type": vision_corruption_type,
        "vision_severities": vision_severities,
        "hard_swap_jaccard_min": hard_swap_jaccard_min,
        "hard_swap_jaccard_max": hard_swap_jaccard_max,
        "include_both_variants": include_both_variants,
        "enable_ood_hard_swap": enable_ood_hard_swap,
        "enforce_template_disjointness": enforce_template_disjointness,
    }
    manifest["distributions"] = {
        "family": _distribution(assigned, key=lambda ex: ex.family.value),
        "operator": _distribution(assigned, key=lambda ex: ex.operator.value),
        "severity": _distribution(assigned, key=lambda ex: str(ex.severity)),
        "split": _distribution(assigned, key=lambda ex: ex.split.value),
    }
    manifest["manifest_json"] = json.dumps(manifest, sort_keys=True)
    return assigned, manifest


def _distribution(examples: list[ConflictExample], key) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for ex in examples:
        out[str(key(ex))] += 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))
