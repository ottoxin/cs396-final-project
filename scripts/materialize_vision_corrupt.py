#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from PIL import Image

from carm.data.io import load_examples, save_examples
from carm.data.schema import CorruptModality, Operator
from carm.data.vision import SEVERITY_AREA_FRACTION, apply_occlusion, occlusion_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize pixel-level vision corruption images and rewrite dataset image paths."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_image_dir", required=True)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fingerprint_images", action="store_true")
    parser.add_argument(
        "--download_missing_coco",
        action="store_true",
        help="Download missing COCO source images for referenced paths under data/raw/coco/{train2014,val2014}.",
    )
    return parser.parse_args()


def _stable_name(example_id: str, base_id: str, variant_id: str) -> str:
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:10]
    safe_base = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in base_id)
    safe_variant = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in variant_id)
    return f"{safe_base}__{safe_variant}__{digest}.jpg"


def _canonicalize_path(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _coco_url_for_local_path(path: Path) -> str | None:
    parts = path.parts
    if "coco" not in parts:
        return None
    idx = parts.index("coco")
    rel = parts[idx + 1 :]
    if len(rel) < 2:
        return None
    split = rel[0]
    file_name = rel[-1]
    if split not in {"train2014", "val2014"}:
        return None
    return f"http://images.cocodataset.org/{split}/{file_name}"


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, filename=str(dst))


def _dir_fingerprint(root: Path) -> dict[str, str | int]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    total_bytes = 0
    for path in files:
        rel = str(path.relative_to(root))
        size = path.stat().st_size
        file_hash = _file_sha256(path)
        total_bytes += size
        h.update(f"{rel}|{size}|{file_hash}\n".encode("utf-8"))
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "fingerprint_sha256": h.hexdigest(),
    }


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input_jsonl)
    workspace_root = Path.cwd()
    out_img_root = Path(args.output_image_dir)
    out_img_root.mkdir(parents=True, exist_ok=True)

    counts = {
        "total_examples": len(examples),
        "vision_candidates": 0,
        "materialized": 0,
        "skipped_existing": 0,
        "skipped_non_occlusion": 0,
        "missing_source_image": 0,
        "downloaded_missing_sources": 0,
        "failed_missing_source_download": 0,
    }
    ensured_sources: set[str] = set()

    for ex in examples:
        if not (ex.operator == Operator.VISION_CORRUPT and ex.corrupt_modality == CorruptModality.VISION):
            continue
        counts["vision_candidates"] += 1

        recipe = ex.metadata.get("vision_recipe", {}) if isinstance(ex.metadata, dict) else {}
        ctype = str(recipe.get("type", "occlusion")).lower()
        severity = int(recipe.get("severity", ex.severity))
        if ctype != "occlusion":
            counts["skipped_non_occlusion"] += 1
            continue

        source_path = Path(ex.image_path)
        source_key = str(source_path)
        if source_key not in ensured_sources and not source_path.exists():
            if args.download_missing_coco:
                url = _coco_url_for_local_path(source_path)
                if url is None:
                    counts["failed_missing_source_download"] += 1
                else:
                    try:
                        _download_file(url, source_path)
                        counts["downloaded_missing_sources"] += 1
                    except Exception:
                        counts["failed_missing_source_download"] += 1
            ensured_sources.add(source_key)

        if not source_path.exists():
            counts["missing_source_image"] += 1
            continue

        out_name = _stable_name(ex.example_id, ex.base_id, ex.variant_id)
        out_path = out_img_root / ex.split.value / ex.family.value / out_name

        with Image.open(source_path) as img:
            width, height = img.size
        box = occlusion_box(width, height, severity=severity, seed_key=ex.example_id)

        if out_path.exists() and not args.overwrite:
            counts["skipped_existing"] += 1
        else:
            width, height, box = apply_occlusion(
                source_path,
                out_path,
                severity=severity,
                seed_key=ex.example_id,
                jpeg_quality=int(args.jpeg_quality),
            )
            counts["materialized"] += 1
            recipe["box_xyxy"] = [int(v) for v in box]
            recipe["image_width"] = int(width)
            recipe["image_height"] = int(height)

        source_store = _canonicalize_path(source_path, workspace_root)
        out_store = _canonicalize_path(out_path, workspace_root)
        recipe["materialized"] = True
        recipe["source_image_path"] = source_store
        recipe["materialized_image_path"] = out_store
        recipe["payload"] = out_store
        ex.metadata["vision_recipe"] = recipe
        ex.image_path = out_store

    save_examples(args.output_jsonl, examples)

    result = {
        "ok": True,
        "counts": counts,
        "output_jsonl": args.output_jsonl,
    }

    manifest_path = args.manifest_json
    if manifest_path:
        manifest = {
            "input_jsonl": args.input_jsonl,
            "input_jsonl_sha256": _file_sha256(args.input_jsonl),
            "output_jsonl": args.output_jsonl,
            "output_jsonl_sha256": _file_sha256(args.output_jsonl),
            "output_image_dir": str(out_img_root),
            "config": {
                "jpeg_quality": int(args.jpeg_quality),
                "overwrite": bool(args.overwrite),
                "download_missing_coco": bool(args.download_missing_coco),
                "deterministic_seed_key": "example_id",
                "vision_type": "occlusion",
                "severity_area_fraction": {str(k): float(v) for k, v in sorted(SEVERITY_AREA_FRACTION.items())},
            },
            "counts": counts,
        }
        if args.fingerprint_images:
            manifest["image_dir_fingerprint"] = _dir_fingerprint(out_img_root)

        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result["manifest_json"] = manifest_path

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
