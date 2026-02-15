from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import replace

from carm.data.integrity import validate_split_integrity
from carm.data.labeling import derive_oracle_action
from carm.data.schema import (
    Action,
    ConflictExample,
    ConflictType,
    CorruptedModality,
    EvidenceModality,
    Split,
)
from carm.data.transforms import caption_swap, typed_text_edit, vision_corruption


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    return len(ta.intersection(tb)) / max(1, len(ta.union(tb)))


def infer_source_image_id(image_path: str) -> str:
    return image_path.split("|")[0]


def infer_template_id(question: str) -> str:
    normalized = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def assign_oracle_labels(example: ConflictExample) -> ConflictExample:
    oracle = derive_oracle_action(example.evidence_modality, example.corrupted_modality)
    return replace(example, oracle_action=oracle)


def adversarial_filter(
    candidates: list[ConflictExample],
    min_similarity: float = 0.18,
    max_similarity: float = 0.72,
) -> list[ConflictExample]:
    """
    Filter out trivial mismatches and near-duplicates to reduce shortcut cues.
    """
    filtered: list[ConflictExample] = []
    for ex in candidates:
        sim = _jaccard(ex.question, ex.text_input)
        if min_similarity <= sim <= max_similarity:
            ex.metadata = {**ex.metadata, "lexical_similarity": round(sim, 4)}
            filtered.append(ex)
    return filtered


def similarity_balanced_sampling(
    candidates: list[ConflictExample],
    bins: int = 4,
    per_bin: int = 100,
    seed: int = 7,
) -> list[ConflictExample]:
    if not candidates:
        return []
    rng = random.Random(seed)
    buckets: dict[int, list[ConflictExample]] = defaultdict(list)
    for ex in candidates:
        sim = float(ex.metadata.get("lexical_similarity", _jaccard(ex.question, ex.text_input)))
        idx = min(bins - 1, int(sim * bins))
        buckets[idx].append(ex)

    sampled: list[ConflictExample] = []
    for idx in range(bins):
        bucket = buckets.get(idx, [])
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_bin])
    return sampled


def _disjoint_split_by_source(
    examples: list[ConflictExample],
    held_out_family: str,
    held_out_severity: int,
    seed: int,
) -> list[ConflictExample]:
    rng = random.Random(seed)
    by_source: dict[str, list[ConflictExample]] = defaultdict(list)
    for ex in examples:
        key = ex.source_image_id or infer_source_image_id(ex.image_path)
        by_source[key].append(ex)

    sources = list(by_source.keys())
    rng.shuffle(sources)
    n = len(sources)
    train_cut = int(n * 0.7)
    val_cut = int(n * 0.85)
    train_sources = set(sources[:train_cut])
    val_sources = set(sources[train_cut:val_cut])
    test_sources = set(sources[val_cut:])

    # Promote entire sources to TEST if any member belongs to a held-out family/severity.
    forced_test_sources: set[str] = set()
    for src, src_examples in by_source.items():
        for ex in src_examples:
            if ex.corruption_family == held_out_family:
                forced_test_sources.add(src)
                break
            if ex.severity >= held_out_severity and ex.corrupted_modality != CorruptedModality.NONE:
                forced_test_sources.add(src)
                break

    train_sources = train_sources.difference(forced_test_sources)
    val_sources = val_sources.difference(forced_test_sources)
    test_sources = test_sources.union(forced_test_sources)

    out: list[ConflictExample] = []
    for src, src_examples in by_source.items():
        for ex in src_examples:
            if src in val_sources:
                split = Split.VAL
            elif src in test_sources:
                split = Split.TEST
            else:
                split = Split.TRAIN
            out.append(replace(ex, split=split))
    return out


def build_conflict_suite(
    clean_examples: list[ConflictExample],
    seed: int = 7,
    held_out_family: str = "text_edit_relation",
    held_out_severity: int = 3,
) -> tuple[list[ConflictExample], dict[str, str]]:
    """
    Construct controlled conflict examples from clean pairs with typed conflicts and split rules.
    """
    rng = random.Random(seed)
    by_topic = defaultdict(list)
    for ex in clean_examples:
        topic = str(ex.metadata.get("topic", "generic"))
        by_topic[topic].append(ex)

    generated: list[ConflictExample] = []
    for ex in clean_examples:
        base = replace(
            ex,
            source_image_id=ex.source_image_id or infer_source_image_id(ex.image_path),
            template_id=ex.template_id or infer_template_id(ex.question),
            split=Split.TRAIN,
            conflict_type=ConflictType.NONE,
            corrupted_modality=CorruptedModality.NONE,
            corruption_family="clean",
            severity=0,
            oracle_action=Action.REQUIRE_AGREEMENT,
        )
        generated.append(assign_oracle_labels(base))

        topic = str(ex.metadata.get("topic", "generic"))
        donors = [d for d in by_topic[topic] if d.example_id != ex.example_id]
        if donors:
            donor = rng.choice(donors)
            generated.append(assign_oracle_labels(caption_swap(base, donor.text_input, same_topic=True, seed=seed)))

        generated.append(assign_oracle_labels(typed_text_edit(base, ConflictType.ATTRIBUTE)))
        generated.append(assign_oracle_labels(typed_text_edit(base, ConflictType.RELATION)))
        generated.append(assign_oracle_labels(typed_text_edit(base, ConflictType.COUNT)))

        for severity in [1, 2, 3]:
            mode = rng.choice(["blur", "occlusion", "distractor"])
            generated.append(assign_oracle_labels(vision_corruption(base, mode=mode, severity=severity)))

    filtered = adversarial_filter(generated)
    balanced = similarity_balanced_sampling(filtered, bins=4, per_bin=max(1, len(filtered) // 8), seed=seed)

    finalized = _disjoint_split_by_source(
        balanced,
        held_out_family=held_out_family,
        held_out_severity=held_out_severity,
        seed=seed,
    )

    # Recompute oracle action after split for deterministic serialization.
    finalized = [assign_oracle_labels(ex) for ex in finalized]
    manifest = validate_split_integrity(finalized)
    return finalized, manifest
