from __future__ import annotations

from collections import defaultdict

from torch.utils.data import Dataset

from carm.data.schema import ConflictExample, CorruptedModality


class ConflictDataset(Dataset[ConflictExample]):
    def __init__(self, examples: list[ConflictExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ConflictExample:
        return self.examples[idx]


def build_clean_index(examples: list[ConflictExample]) -> dict[str, ConflictExample]:
    """Index clean reference examples for counterfactual pairs."""
    idx: dict[str, ConflictExample] = {}
    for ex in examples:
        if ex.corrupted_modality != CorruptedModality.NONE:
            continue
        key = pair_key(ex)
        idx[key] = ex
    return idx


def pair_key(ex: ConflictExample) -> str:
    src = ex.source_image_id or ex.image_path.split("|")[0]
    return f"{src}::{ex.question}"


def group_by_corruption_family(examples: list[ConflictExample]) -> dict[str, list[ConflictExample]]:
    out: dict[str, list[ConflictExample]] = defaultdict(list)
    for ex in examples:
        out[ex.corruption_family].append(ex)
    return out
