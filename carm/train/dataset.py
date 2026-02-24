from __future__ import annotations

from collections import defaultdict

from torch.utils.data import Dataset

from carm.data.schema import ConflictExample, CorruptModality


class ConflictDataset(Dataset[ConflictExample]):
    def __init__(self, examples: list[ConflictExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ConflictExample:
        return self.examples[idx]


def build_clean_index(examples: list[ConflictExample]) -> dict[str, ConflictExample]:
    idx: dict[str, ConflictExample] = {}
    for ex in examples:
        if ex.corrupt_modality != CorruptModality.NONE:
            continue
        idx[pair_key(ex)] = ex
    return idx


def pair_key(ex: ConflictExample) -> str:
    src = ex.source_image_id or ex.image_path.split("|")[0]
    return f"{src}::{ex.question}"


def group_by_operator(examples: list[ConflictExample]) -> dict[str, list[ConflictExample]]:
    out: dict[str, list[ConflictExample]] = defaultdict(list)
    for ex in examples:
        out[ex.operator.value].append(ex)
    return out
