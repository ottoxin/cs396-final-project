from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from carm.data.schema import ConflictExample, Operator


def _group_by_base(examples: list[ConflictExample]) -> dict[str, list[ConflictExample]]:
    grouped: dict[str, list[ConflictExample]] = defaultdict(list)
    for ex in examples:
        grouped[ex.base_id].append(ex)
    return grouped


def _representative(group: list[ConflictExample]) -> ConflictExample:
    for ex in group:
        if ex.operator == Operator.CLEAN:
            return ex
    return sorted(group, key=lambda e: e.example_id)[0]


def _allocate_counts(total: int, sizes: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    if total <= 0:
        return {k: 0 for k in sizes}
    n = sum(sizes.values())
    if n == 0:
        return {k: 0 for k in sizes}

    base_alloc: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []

    for key, size in sizes.items():
        raw = (size / n) * total
        cnt = int(raw)
        base_alloc[key] = cnt
        fractions.append((raw - cnt, key))

    assigned = sum(base_alloc.values())
    remaining = max(0, total - assigned)

    for _, key in sorted(fractions, key=lambda kv: kv[0], reverse=True)[:remaining]:
        base_alloc[key] += 1

    for key, size in sizes.items():
        base_alloc[key] = min(base_alloc[key], size)

    assigned = sum(base_alloc.values())
    if assigned < total:
        deficit = total - assigned
        for key, size in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True):
            room = size - base_alloc[key]
            if room <= 0:
                continue
            take = min(room, deficit)
            base_alloc[key] += take
            deficit -= take
            if deficit == 0:
                break

    return base_alloc


def _distribution(examples: list[ConflictExample], key_fn) -> dict[str, int]:
    counter = Counter(key_fn(ex) for ex in examples)
    return dict(sorted(((str(k), int(v)) for k, v in counter.items()), key=lambda kv: kv[0]))


def sample_pilot_by_base(
    examples: list[ConflictExample],
    *,
    base_sample_size: int,
    seed: int,
) -> tuple[list[ConflictExample], dict[str, Any]]:
    grouped = _group_by_base(examples)
    reps: dict[str, ConflictExample] = {base_id: _representative(group) for base_id, group in grouped.items()}

    if base_sample_size <= 0:
        return [], {
            "base_sample_size": 0,
            "selected_base_count": 0,
            "selected_example_count": 0,
        }

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for base_id, rep in reps.items():
        key = (rep.family.value, rep.split.value)
        strata[key].append(base_id)

    for ids in strata.values():
        ids.sort()

    sizes = {k: len(v) for k, v in strata.items()}
    alloc = _allocate_counts(min(base_sample_size, len(grouped)), sizes)

    rng = random.Random(seed)
    selected_base_ids: set[str] = set()
    for key, ids in strata.items():
        count = alloc.get(key, 0)
        shuffled = list(ids)
        rng.shuffle(shuffled)
        selected_base_ids.update(shuffled[:count])

    selected = [ex for ex in examples if ex.base_id in selected_base_ids]
    selected = sorted(selected, key=lambda ex: ex.example_id)

    manifest = {
        "strategy": "stratified_base",
        "seed": seed,
        "base_sample_size": base_sample_size,
        "selected_base_count": len(selected_base_ids),
        "selected_example_count": len(selected),
        "strata_counts_full": {f"{k[0]}::{k[1]}": len(v) for k, v in sorted(strata.items())},
        "strata_counts_selected": {
            f"{k[0]}::{k[1]}": sum(1 for base_id in selected_base_ids if base_id in set(v))
            for k, v in sorted(strata.items())
        },
        "distributions": {
            "family": _distribution(selected, key_fn=lambda ex: ex.family.value),
            "operator": _distribution(selected, key_fn=lambda ex: ex.operator.value),
            "severity": _distribution(selected, key_fn=lambda ex: ex.severity),
            "split": _distribution(selected, key_fn=lambda ex: ex.split.value),
        },
    }
    return selected, manifest
