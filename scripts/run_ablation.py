#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation train+eval workflow.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    train_dir = out / "train"
    eval_dir = out / "eval"

    subprocess.run(
        [
            "python3",
            "scripts/train_carm.py",
            "--config",
            args.config,
            "--train_jsonl",
            args.input_jsonl,
            "--output_dir",
            str(train_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            "python3",
            "scripts/evaluate_carm.py",
            "--config",
            args.config,
            "--input_jsonl",
            args.input_jsonl,
            "--output_dir",
            str(eval_dir),
            "--model_ckpt",
            str(train_dir / "carm_heads.pt"),
            "--split",
            "test_id",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
