#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carm.data.answer_vocab import build_family_vocabs, save_family_vocabs
from carm.data.io import load_examples
from carm.data.schema import Split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-derived family vocab from prepared examples.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--split",
        default=Split.TRAIN.value,
        choices=[s.value for s in Split],
        help="Examples from this split contribute to the vocab. Defaults to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input_jsonl)
    target_split = Split(args.split)
    selected = [ex for ex in examples if ex.split == target_split]
    vocabs = build_family_vocabs(selected)
    save_family_vocabs(vocabs, args.output_json)
    print(f"built family vocab from {len(selected)} {target_split.value} examples")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
