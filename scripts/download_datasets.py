#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
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
DEFAULT_HF_RELEASE_REPO = "haohxin/cs396-final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download datasets for CARM. "
            "Use --source official for raw VQAv2/COCO or --source release for a prebuilt HF release snapshot."
        )
    )
    parser.add_argument(
        "--source",
        default="official",
        choices=["official", "release"],
        help="Dataset source to download.",
    )
    parser.add_argument(
        "--root",
        default="data/raw",
        help=(
            "Download root directory for --source official. "
            "Also used by --with-official-images in --source release mode."
        ),
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip COCO image archives (default is to include images).",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=DEFAULT_HF_RELEASE_REPO,
        help="Hugging Face dataset repo id for --source release (for example org/repo).",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help="Revision/tag/branch for --source release.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token for private datasets.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional cache directory for HF downloads.",
    )
    parser.add_argument(
        "--release-cache-root",
        default="data/hf_release/cs396-final-dataset",
        help="Local directory to store raw downloaded release files.",
    )
    parser.add_argument(
        "--install-data-root",
        default="data",
        help="Project data root where release artifacts will be installed.",
    )
    parser.add_argument(
        "--skip-release-images",
        action="store_true",
        help="Skip pulling and installing vision_corrupt images for --source release.",
    )
    parser.add_argument(
        "--with-official-images",
        action="store_true",
        help=(
            "For --source release, also download official COCO train/val images into --root "
            "(needed for clean/swap/text rows in pilot JSONL)."
        ),
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


def _download_official(args: argparse.Namespace) -> None:
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


def _download_official_images_only(root: Path) -> None:
    archives_dir = root / "archives"
    tasks = [
        ("coco_images_train", root / "coco"),
        ("coco_images_val", root / "coco"),
    ]
    manifest: dict[str, str] = {}
    for key, extract_dir in tasks:
        url = DOWNLOADS[key]
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = archives_dir / archive_name
        _download(url, archive_path)
        _extract(archive_path, extract_dir)
        manifest[key] = str(archive_path)
    manifest_path = root / "download_manifest_images_only.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote images-only manifest: {manifest_path}")


def _require_hf_download_support() -> object:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'huggingface_hub' for --source release. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    return snapshot_download


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"installed file: {dst}")


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"skip missing directory: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for child in src.rglob("*"):
        if child.is_dir():
            continue
        rel = child.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, out_path)
        copied += 1
    print(f"installed {copied} files into: {dst}")


def _find_release_file(release_root: Path, filename: str) -> Path | None:
    direct = release_root / filename
    if direct.exists():
        return direct
    matches = sorted(p for p in release_root.rglob(filename) if p.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"warning: multiple matches for {filename}; using {matches[0]}",
            file=sys.stderr,
        )
    return matches[0]


def _find_release_dir(release_root: Path, rel_dir: str) -> Path | None:
    direct = release_root / rel_dir
    if direct.exists() and direct.is_dir():
        return direct
    parts = Path(rel_dir).parts
    matches: list[Path] = []
    for candidate in release_root.rglob(parts[-1]):
        if not candidate.is_dir():
            continue
        if candidate.parts[-len(parts) :] == parts:
            matches.append(candidate)
    if not matches:
        return None
    matches.sort()
    if len(matches) > 1:
        print(
            f"warning: multiple matches for {rel_dir}; using {matches[0]}",
            file=sys.stderr,
        )
    return matches[0]


def _rewrite_manifest_paths(manifest_path: Path, data_root: Path) -> None:
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pilot_base = data_root / "generated/pilots/pilot_3k_class_medium.jsonl"
    pilot_real = data_root / "generated/pilots/pilot_3k_class_medium_real_vision.jsonl"
    image_dir = data_root / "generated/vision_corrupt/class_medium/pilot_3k"
    if pilot_base.exists():
        data["input_jsonl"] = str(pilot_base)
    else:
        data["input_jsonl"] = str(pilot_real)
        note = (
            "base pilot jsonl was not included in the release snapshot; "
            "input_jsonl is set to output_jsonl for local path validity"
        )
        notes = data.get("notes")
        if isinstance(notes, list):
            notes.append(note)
        else:
            data["notes"] = [note]
    data["output_jsonl"] = str(pilot_real)
    data["output_image_dir"] = str(image_dir)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"normalized manifest paths: {manifest_path}")


def _install_release_artifacts(release_root: Path, data_root: Path, include_images: bool) -> None:
    file_map = {
        "pilot_3k_class_medium_real_vision.jsonl": data_root / "generated/pilots/pilot_3k_class_medium_real_vision.jsonl",
        "pilot_3k_class_medium_real_vision.manifest.json": data_root
        / "generated/pilots/pilot_3k_class_medium_real_vision.manifest.json",
        "pilot_3k_class_medium.manifest.json": data_root / "generated/pilots/pilot_3k_class_medium.manifest.json",
        "conflict_suite_class_medium.manifest.json": data_root / "generated/conflict_suite_class_medium.manifest.json",
        "base_examples_class_medium.stats.json": data_root / "interim/base_examples_class_medium.stats.json",
    }
    for rel_src, dst in file_map.items():
        src = _find_release_file(release_root, rel_src)
        if src:
            _copy_file(src, dst)
        else:
            print(f"warning: expected release file missing: {rel_src}", file=sys.stderr)

    if include_images:
        image_src = _find_release_dir(release_root, "vision_corrupt/class_medium/pilot_3k")
        if image_src is None:
            print(
                "warning: expected release directory missing: vision_corrupt/class_medium/pilot_3k",
                file=sys.stderr,
            )
        else:
            _copy_tree(
                image_src,
                data_root / "generated/vision_corrupt/class_medium/pilot_3k",
            )

    _rewrite_manifest_paths(
        data_root / "generated/pilots/pilot_3k_class_medium_real_vision.manifest.json",
        data_root,
    )


def _download_release(args: argparse.Namespace) -> None:
    snapshot_download = _require_hf_download_support()

    release_root = Path(args.release_cache_root)
    release_root.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        "pilot_3k_class_medium_real_vision.jsonl",
        "pilot_3k_class_medium_real_vision.manifest.json",
        "pilot_3k_class_medium.manifest.json",
        "conflict_suite_class_medium.manifest.json",
        "base_examples_class_medium.stats.json",
        "README.md",
        "**/pilot_3k_class_medium_real_vision.jsonl",
        "**/pilot_3k_class_medium_real_vision.manifest.json",
        "**/pilot_3k_class_medium.manifest.json",
        "**/conflict_suite_class_medium.manifest.json",
        "**/base_examples_class_medium.stats.json",
        "**/README.md",
    ]
    if not args.skip_release_images:
        allow_patterns.append("vision_corrupt/class_medium/pilot_3k/**")
        allow_patterns.append("**/vision_corrupt/class_medium/pilot_3k/**")

    print(f"download release snapshot: repo={args.hf_repo_id} revision={args.hf_revision}")
    snapshot_download(
        repo_id=args.hf_repo_id,
        repo_type="dataset",
        revision=args.hf_revision,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
        local_dir=str(release_root),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )

    data_root = Path(args.install_data_root)
    _install_release_artifacts(
        release_root=release_root,
        data_root=data_root,
        include_images=not args.skip_release_images,
    )
    install_manifest = {
        "source": "release",
        "hf_repo_id": args.hf_repo_id,
        "hf_revision": args.hf_revision,
        "release_cache_root": str(release_root),
        "install_data_root": str(data_root),
        "include_images": not args.skip_release_images,
        "with_official_images": args.with_official_images,
        "official_root": args.root,
    }
    manifest_path = data_root / "download_manifest_release.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(install_manifest, indent=2), encoding="utf-8")
    print(f"wrote release install manifest: {manifest_path}")

    if args.with_official_images:
        _download_official_images_only(Path(args.root))
    else:
        print(
            "note: release pull does not include original COCO train/val images. "
            "For full pilot inference, rerun with --with-official-images "
            "or run --source official separately.",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    if args.source == "official":
        _download_official(args)
    else:
        _download_release(args)


if __name__ == "__main__":
    main()
