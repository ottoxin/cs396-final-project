#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from carm.data.integrity import validate_split_integrity
from carm.data.io import load_examples
from carm.data.schema import Family
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset split integrity.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input_jsonl)

    heldout_family = None
    heldout_severity = 3
    enforce_template_disjointness = False
    if args.config:
        cfg = load_yaml_config(args.config)
        split_cfg = cfg.get("splits", {})
        heldout_family = Family(str(split_cfg.get("ood_family", Family.ATTRIBUTE_COLOR.value)))
        heldout_severity = int(split_cfg.get("ood_severity", 3))
        enforce_template_disjointness = bool(split_cfg.get("enforce_template_disjointness", False))

    manifest = validate_split_integrity(
        examples,
        heldout_family=heldout_family,
        heldout_severity=heldout_severity,
        enforce_template_disjointness=enforce_template_disjointness,
    )
    print(json.dumps({"ok": True, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
