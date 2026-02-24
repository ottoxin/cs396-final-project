#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carm.data.io import save_examples
from carm.data.schema import Family
from carm.data.vqa_coco import build_base_examples
from carm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean base dataset from official VQAv2+COCO JSON files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--stats_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    data_cfg = cfg.get("data", {})
    paths_cfg = data_cfg.get("paths", {})

    families = [Family(str(f)) for f in data_cfg.get("families", ["existence", "count", "attribute_color"])]

    examples, stats = build_base_examples(
        vqa_questions_train=str(paths_cfg["vqa_questions_train"]),
        vqa_questions_val=str(paths_cfg["vqa_questions_val"]),
        vqa_annotations_train=str(paths_cfg["vqa_annotations_train"]),
        vqa_annotations_val=str(paths_cfg["vqa_annotations_val"]),
        coco_captions_train=str(paths_cfg["coco_captions_train"]),
        coco_captions_val=str(paths_cfg["coco_captions_val"]),
        image_train_dir=str(paths_cfg["coco_images_train_dir"]),
        image_val_dir=str(paths_cfg["coco_images_val_dir"]),
        families=families,
        color_vocab=list(data_cfg.get("color_vocab", [])),
        consistency_filter=bool(data_cfg.get("consistency_filter", True)),
        seed=int(cfg.get("seed", 7)),
        max_per_family=(
            int(data_cfg["max_per_family"])
            if data_cfg.get("max_per_family") is not None
            else None
        ),
    )

    output_jsonl = args.output_jsonl or str(paths_cfg.get("base_examples_jsonl", "data/interim/base_examples.jsonl"))
    save_examples(output_jsonl, examples)

    stats_path = args.stats_json or str(Path(output_jsonl).with_suffix(".stats.json"))
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"wrote {len(examples)} base examples to {output_jsonl}")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
