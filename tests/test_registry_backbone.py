from __future__ import annotations

import unittest

from carm.models.backbone import MockFrozenBackbone, Qwen25VLAdapter
from carm.models.registry import create_backbone


class TestBackboneRegistry(unittest.TestCase):
    def test_create_mock_backbone_default(self) -> None:
        backbone = create_backbone({})
        self.assertIsInstance(backbone, MockFrozenBackbone)

    def test_create_qwen_backbone_uses_registry_model_name(self) -> None:
        cfg = {
            "name": "qwen2_5_vl_7b",
            "registry": {
                "qwen2_5_vl_7b": {
                    "model_name": "Qwen/Test-Model",
                }
            },
        }
        backbone = create_backbone(cfg)
        self.assertIsInstance(backbone, Qwen25VLAdapter)
        self.assertEqual(backbone.model_name, "Qwen/Test-Model")

    def test_create_qwen_backbone_prefers_direct_model_name(self) -> None:
        cfg = {
            "name": "qwen2_5_vl_7b",
            "model_name": "Qwen/Direct-Model",
            "registry": {
                "qwen2_5_vl_7b": {
                    "model_name": "Qwen/Ignored-Model",
                }
            },
        }
        backbone = create_backbone(cfg)
        self.assertIsInstance(backbone, Qwen25VLAdapter)
        self.assertEqual(backbone.model_name, "Qwen/Direct-Model")


if __name__ == "__main__":
    unittest.main()
