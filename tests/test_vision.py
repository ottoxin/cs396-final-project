from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from carm.data.vision import apply_occlusion, occlusion_box


class TestVisionCorrupt(unittest.TestCase):
    def test_occlusion_box_deterministic(self) -> None:
        a = occlusion_box(640, 480, severity=2, seed_key="ex-1")
        b = occlusion_box(640, 480, severity=2, seed_key="ex-1")
        self.assertEqual(a, b)

    def test_apply_occlusion_writes_image(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src.jpg"
            dst = root / "dst.jpg"

            Image.new("RGB", (200, 100), (255, 255, 255)).save(src, format="JPEG")
            w, h, box = apply_occlusion(src, dst, severity=3, seed_key="ex-2")
            self.assertEqual((w, h), (200, 100))
            self.assertTrue(dst.exists())

            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            with Image.open(dst) as out:
                pixel = out.convert("RGB").getpixel((cx, cy))
            # JPEG compression may shift exact value, but center should be very dark.
            self.assertLessEqual(max(pixel), 40)


if __name__ == "__main__":
    unittest.main()
