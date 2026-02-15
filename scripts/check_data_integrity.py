#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from carm.data.integrity import validate_split_integrity
from carm.data.io import load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset split integrity.")
    parser.add_argument("--input_jsonl", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input_jsonl)
    manifest = validate_split_integrity(examples)
    print(json.dumps({"ok": True, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
