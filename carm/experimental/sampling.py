from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from carm.data.schema import ConflictExample, Split
from carm.experimental.labels import DerivedLabels


@dataclass(frozen=True)
class SmallRunConfig:
    max_train_examples: int
    max_val_examples: int
    max_test_examples: int
    random_seed: int
    sampling_strategy: str
    run_name: str


def _sampling_key(example: ConflictExample, derived: DerivedLabels, strategy: str) -> str:
    if strategy == "random":
        return "__all__"
    if strategy == "joint_info_state":
        return derived.joint_info_state or derived.derivation_status
    if strategy == "joint_info_state_family":
        return "::".join(
            [
                derived.joint_info_state or derived.derivation_status,
                example.family.value,
            ]
        )
    if strategy == "pairwise_relation":
        return derived.pairwise_relation or derived.derivation_status
    if strategy == "pairwise_relation_family":
        return "::".join(
            [
                derived.pairwise_relation or derived.derivation_status,
                example.family.value,
            ]
        )
    if strategy == "protocol_category_family":
        protocol_category = ""
        if isinstance(example.metadata, dict):
            protocol_category = str(example.metadata.get("protocol_category", "")).strip()
        return "::".join(
            [
                protocol_category or derived.pairwise_relation or derived.derivation_status,
                example.family.value,
            ]
        )
    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def _round_robin_sample(
    examples: list[ConflictExample],
    derived_by_id: dict[str, DerivedLabels],
    *,
    max_examples: int,
    strategy: str,
    seed: int,
    split_name: str,
) -> tuple[list[ConflictExample], dict[str, Any]]:
    if max_examples <= 0:
        return [], {
            "strategy": strategy,
            "fallback_behavior": "requested_zero_examples",
            "source_size": len(examples),
            "selected_size": 0,
        }
    if max_examples >= len(examples):
        return list(examples), {
            "strategy": strategy,
            "fallback_behavior": "full_split_no_sampling",
            "source_size": len(examples),
            "selected_size": len(examples),
        }

    rng = random.Random(f"{seed}:{split_name}:{strategy}")
    groups: dict[str, list[ConflictExample]] = defaultdict(list)
    for example in examples:
        groups[_sampling_key(example, derived_by_id[example.example_id], strategy)].append(example)

    for key, members in groups.items():
        ordered = list(members)
        rng.shuffle(ordered)
        groups[key] = ordered

    group_order = sorted(groups)
    fallback_behavior = "balanced_round_robin"
    if len(group_order) > max_examples:
        fallback_behavior = "partial_strata_round_robin"
        rng.shuffle(group_order)

    selected: list[ConflictExample] = []
    used_counts: Counter[str] = Counter()
    while len(selected) < max_examples:
        progressed = False
        for key in group_order:
            if len(selected) >= max_examples:
                break
            members = groups[key]
            if not members:
                continue
            selected.append(members.pop())
            used_counts[key] += 1
            progressed = True
        if not progressed:
            break

    selected_ids = {example.example_id for example in selected}
    selected_label_counts: Counter[str] = Counter()
    selected_family_counts: Counter[str] = Counter()
    for example in selected:
        derived = derived_by_id[example.example_id]
        selected_label_counts[_sampling_key(example, derived, strategy)] += 1
        selected_family_counts[example.family.value] += 1

    return selected, {
        "strategy": strategy,
        "fallback_behavior": fallback_behavior,
        "source_size": len(examples),
        "selected_size": len(selected),
        "group_count": len(group_order),
        "group_allocation_counts": dict(sorted(used_counts.items(), key=lambda item: item[0])),
        "selected_sampling_label_counts": dict(sorted(selected_label_counts.items(), key=lambda item: item[0])),
        "selected_family_counts": dict(sorted(selected_family_counts.items(), key=lambda item: item[0])),
        "selected_example_ids": [example.example_id for example in selected],
        "omitted_example_count": len(examples) - len(selected_ids),
    }


def build_small_run_splits(
    examples: list[ConflictExample],
    derived_by_id: dict[str, DerivedLabels],
    cfg: SmallRunConfig,
) -> tuple[dict[str, list[ConflictExample]], dict[str, Any]]:
    by_split: dict[Split, list[ConflictExample]] = defaultdict(list)
    for example in examples:
        by_split[example.split].append(example)

    split_limits = {
        Split.TRAIN: int(cfg.max_train_examples),
        Split.VAL: int(cfg.max_val_examples),
        Split.TEST_ID: int(cfg.max_test_examples),
    }

    selected: dict[str, list[ConflictExample]] = {}
    manifest: dict[str, Any] = {
        "run_name": cfg.run_name,
        "random_seed": int(cfg.random_seed),
        "sampling_strategy": cfg.sampling_strategy,
        "requested_limits": {
            "train": int(cfg.max_train_examples),
            "val": int(cfg.max_val_examples),
            "test_id": int(cfg.max_test_examples),
        },
        "split_manifests": {},
    }

    for split in (Split.TRAIN, Split.VAL, Split.TEST_ID):
        sampled, split_manifest = _round_robin_sample(
            by_split.get(split, []),
            derived_by_id,
            max_examples=split_limits[split],
            strategy=cfg.sampling_strategy,
            seed=int(cfg.random_seed),
            split_name=split.value,
        )
        selected[split.value] = sampled
        manifest["split_manifests"][split.value] = split_manifest

    return selected, manifest
