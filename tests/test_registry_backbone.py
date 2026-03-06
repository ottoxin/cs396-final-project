from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from carm.data.schema import Family
from carm.models.backbone import Qwen25VLAdapter
from carm.models.registry import create_backbone


class TestBackboneRegistry(unittest.TestCase):
    def test_create_qwen_backbone_default(self) -> None:
        backbone = create_backbone({})
        self.assertIsInstance(backbone, Qwen25VLAdapter)

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

    def test_create_qwen_backbone_passes_extended_generation_config(self) -> None:
        cfg = {
            "name": "qwen2_5_vl_7b",
            "max_new_tokens": 7,
            "count_min": 2,
            "count_max": 5,
            "color_vocab": ["red", "gray"],
        }

        backbone = create_backbone(cfg)

        self.assertIsInstance(backbone, Qwen25VLAdapter)
        self.assertEqual(backbone.config.max_new_tokens, 7)
        self.assertEqual(backbone.config.count_min, 2)
        self.assertEqual(backbone.config.count_max, 5)
        self.assertEqual(backbone.config.color_vocab, ("red", "gray"))

    def test_create_qwen_backbone_loads_family_vocab_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "family_vocab.json"
            path.write_text(
                json.dumps(
                    {
                        "existence": ["yes", "no", "unknown"],
                        "count": ["1", "4", "unknown"],
                        "attribute_color": ["beige", "gray", "unknown"],
                    }
                ),
                encoding="utf-8",
            )

            backbone = create_backbone(
                {
                    "name": "qwen2_5_vl_7b",
                    "family_vocab_path": str(path),
                    "force_fallback_distribution": True,
                }
            )

        self.assertIsInstance(backbone, Qwen25VLAdapter)
        self.assertEqual(backbone.config.family_vocab_overrides["count"], ("1", "4", "unknown"))
        self.assertEqual(backbone.config.family_vocab_overrides["attribute_color"], ("beige", "gray", "unknown"))
        self.assertTrue(backbone.config.force_fallback_distribution)
        self.assertEqual(backbone._family_vocab(Family.COUNT), ("1", "4", "unknown"))

    def test_inline_family_vocab_overrides_take_precedence(self) -> None:
        backbone = create_backbone(
            {
                "name": "qwen2_5_vl_7b",
                "family_vocab_overrides": {
                    "count": ["2", "9", "unknown"],
                },
            }
        )

        self.assertEqual(backbone._family_vocab(Family.COUNT), ("2", "9", "unknown"))

    def test_force_fallback_distribution_bypasses_projection(self) -> None:
        backbone = create_backbone(
            {
                "name": "qwen2_5_vl_7b",
                "force_fallback_distribution": True,
            }
        )

        dist = backbone._distribution_from_first_token_logits(torch.randn(32), Family.EXISTENCE)

        self.assertIsNone(dist)

    def test_family_vocab_token_ids_use_direct_last_token_when_unique(self) -> None:
        class _FakeTokenizer:
            mapping = {
                "1": [16],
                "2": [17],
                "3": [18],
            }

            def encode(self, text: str, add_special_tokens: bool = False):
                return list(self.mapping.get(text, []))

        backbone = create_backbone({})
        assert isinstance(backbone, Qwen25VLAdapter)
        backbone._tokenizer = _FakeTokenizer()
        backbone._ensure_loaded = lambda: None  # type: ignore[assignment]

        token_ids = backbone._token_ids_for_vocab(("1", "2", "3"))
        self.assertEqual(token_ids, [16, 17, 18])

    def test_family_vocab_token_ids_fall_back_to_context_when_direct_ids_collide(self) -> None:
        class _FakeTokenizer:
            mapping = {
                "red": [99],
                "blue": [99],
                "Answer: red": [101, 11],
                "Answer: blue": [101, 12],
            }

            def encode(self, text: str, add_special_tokens: bool = False):
                return list(self.mapping.get(text, []))

        backbone = create_backbone({})
        assert isinstance(backbone, Qwen25VLAdapter)
        backbone._tokenizer = _FakeTokenizer()
        backbone._ensure_loaded = lambda: None  # type: ignore[assignment]

        token_ids = backbone._token_ids_for_vocab(("red", "blue"))
        self.assertEqual(token_ids, [11, 12])


if __name__ == "__main__":
    unittest.main()
