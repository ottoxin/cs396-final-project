#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DOWNLOADS = {
    "vqa_questions_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "vqa_questions_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "vqa_annotations_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "vqa_annotations_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "coco_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "coco_images_train": "http://images.cocodataset.org/zips/train2014.zip",
    "coco_images_val": "http://images.cocodataset.org/zips/val2014.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official VQAv2/COCO artifacts using Python stdlib.")
    parser.add_argument("--root", default="data/raw", help="Download root directory")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip COCO image archives (default is to include images).",
    )
    return parser.parse_args()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"skip existing: {dst}")
        return
    print(f"download: {url}")
    urllib.request.urlretrieve(url, filename=str(dst))


def _extract(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
        return
    if archive_path.suffix in {".tar", ".gz", ".tgz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(out_dir)
        return
    raise ValueError(f"Unsupported archive type: {archive_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    archives_dir = root / "archives"

    tasks = [
        ("vqa_questions_train", root / "vqa"),
        ("vqa_questions_val", root / "vqa"),
        ("vqa_annotations_train", root / "vqa"),
        ("vqa_annotations_val", root / "vqa"),
        ("coco_annotations", root / "coco"),
    ]

    if not args.no_images:
        tasks.extend(
            [
                ("coco_images_train", root / "coco"),
                ("coco_images_val", root / "coco"),
            ]
        )

    manifest: dict[str, str] = {}
    for key, extract_dir in tasks:
        url = DOWNLOADS[key]
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = archives_dir / archive_name
        _download(url, archive_path)
        _extract(archive_path, extract_dir)
        manifest[key] = str(archive_path)

    manifest_path = root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote download manifest: {manifest_path}")


if __name__ == "__main__":
    main()
