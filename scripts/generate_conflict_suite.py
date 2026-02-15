#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carm.data.construction import build_conflict_suite
from carm.data.io import load_examples, save_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Conflict Suite from clean JSONL.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--held_out_family", default="text_edit_relation")
    parser.add_argument("--held_out_severity", type=int, default=3)
    parser.add_argument("--manifest_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean = load_examples(args.input_jsonl)
    suite, manifest = build_conflict_suite(
        clean_examples=clean,
        seed=args.seed,
        held_out_family=args.held_out_family,
        held_out_severity=args.held_out_severity,
    )
    save_examples(args.output_jsonl, suite)

    manifest_path = args.manifest_json
    if manifest_path is None:
        manifest_path = str(Path(args.output_jsonl).with_suffix(".manifest.json"))
    with Path(manifest_path).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {len(suite)} examples to {args.output_jsonl}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
