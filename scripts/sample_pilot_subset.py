#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carm.data.io import load_examples, save_examples
from carm.data.sampling import sample_pilot_by_base
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample deterministic stratified pilot from full canonical suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--manifest_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    examples = load_examples(args.input_jsonl)

    pilot_cfg = cfg.get("sampling", {}).get("pilot", {})
    pilot_size = int(pilot_cfg.get("size", 3000))
    pilot_seed = int(pilot_cfg.get("seed", cfg.get("seed", 7)))

    sampled, manifest = sample_pilot_by_base(
        examples,
        base_sample_size=pilot_size,
        seed=pilot_seed,
    )
    save_examples(args.output_jsonl, sampled)

    manifest_path = args.manifest_json
    if manifest_path is None:
        manifest_path = str(Path(args.output_jsonl).with_suffix(".manifest.json"))
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {len(sampled)} pilot examples to {args.output_jsonl}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
