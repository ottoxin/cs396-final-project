from __future__ import annotations

import json
from pathlib import Path

from carm.data.schema import ConflictExample


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_examples(path: str | Path) -> list[ConflictExample]:
    return [ConflictExample.from_dict(r) for r in read_jsonl(path)]


def save_examples(path: str | Path, examples: list[ConflictExample]) -> None:
    write_jsonl(path, [e.to_dict() for e in examples])
