from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


# Area fraction of the occlusion block by severity.
SEVERITY_AREA_FRACTION = {
    1: 0.15,
    2: 0.30,
    3: 0.45,
}


def _seed_from_key(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def occlusion_box(
    width: int,
    height: int,
    *,
    severity: int,
    seed_key: str,
) -> tuple[int, int, int, int]:
    frac = SEVERITY_AREA_FRACTION.get(int(severity), SEVERITY_AREA_FRACTION[3] if severity > 3 else SEVERITY_AREA_FRACTION[1])
    side_scale = math.sqrt(max(1e-6, min(0.95, frac)))

    box_w = max(1, int(width * side_scale))
    box_h = max(1, int(height * side_scale))
    box_w = min(box_w, width)
    box_h = min(box_h, height)

    rng = random.Random(_seed_from_key(seed_key))
    x0 = rng.randint(0, max(0, width - box_w))
    y0 = rng.randint(0, max(0, height - box_h))
    x1 = min(width, x0 + box_w)
    y1 = min(height, y0 + box_h)
    return x0, y0, x1, y1


def apply_occlusion(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    severity: int,
    seed_key: str,
    fill_rgb: tuple[int, int, int] = (0, 0, 0),
    jpeg_quality: int = 90,
) -> tuple[int, int, tuple[int, int, int, int]]:
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        box = occlusion_box(w, h, severity=severity, seed_key=seed_key)
        draw = ImageDraw.Draw(rgb)
        draw.rectangle(box, fill=fill_rgb)

        ext = dst.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            rgb.save(dst, format="JPEG", quality=int(jpeg_quality), optimize=True)
        elif ext == ".png":
            rgb.save(dst, format="PNG", optimize=True)
        else:
            # Default to JPEG for compact storage.
            dst = dst.with_suffix(".jpg")
            rgb.save(dst, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return w, h, box
