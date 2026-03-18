from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from carm.data.schema import Family
from carm.models.backbone import BackboneConfig, Qwen25VLAdapter


@unittest.skipUnless(
    os.environ.get("RUN_QWEN_INFERENCE_TESTS") == "1",
    "set RUN_QWEN_INFERENCE_TESTS=1 to run real Qwen inference tests",
)
class TestQwenInferenceOptIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        cls.adapter = Qwen25VLAdapter(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            config=BackboneConfig(hidden_size=128, seq_len=32),
            device=device,
            torch_dtype=dtype,
            cache_results=False,
        )
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.multimodal_example = cls._load_example()

    @classmethod
    def _load_example(cls) -> dict[str, str]:
        candidates = [
            cls.repo_root / "data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.jsonl",
            cls.repo_root / "data/cache/hf_5way/prepared/carm_vqa_5way.jsonl",
        ]
        for path in candidates:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    image_path = Path(str(row.get("image_path", "")))
                    if image_path.exists():
                        return {
                            "image_path": str(image_path),
                            "text_input": str(row["text_input"]),
                            "question": str(row["question"]),
                            "family": str(row.get("family", "")),
                        }
        raise FileNotFoundError("No prepared example with an existing image path was found for multimodal Qwen opt-in testing.")

    def _assert_answer_shape(self, answer_text: str, family_value: str) -> None:
        if family_value == Family.EXISTENCE.value:
            self.assertIn(answer_text, {"yes", "no", "unknown"})
            return
        if family_value == Family.COUNT.value:
            self.assertRegex(answer_text, r"^(unknown|\d+)$")
            return
        if family_value == Family.ATTRIBUTE_COLOR.value:
            valid_colors = set(self.adapter.config.color_vocab) | {"unknown"}
            self.assertIn(answer_text, valid_colors)
            return
        self.assertTrue(answer_text)

    def test_text_probe_runs_real_model_with_family_specific_parsing(self) -> None:
        adapter = self.adapter

        existence = adapter.run_probe_text_only(
            text="A bicycle is next to a tree.",
            question="Is there a bicycle in the image?",
        )
        self.assertIn(existence.answer_text, {"yes", "no", "unknown"})
        self.assertTrue(existence.raw_text)
        self.assertGreater(existence.answer_dist.numel(), 0)
        self.assertEqual(tuple(existence.features.shape), (3,))

        count = adapter.run_probe_text_only(
            text="Two dogs sit on grass.",
            question="How many dogs are there?",
        )
        self.assertRegex(count.answer_text, r"^(unknown|\d+)$")
        self.assertTrue(count.raw_text)
        self.assertGreater(count.answer_dist.numel(), 0)
        self.assertEqual(tuple(count.features.shape), (3,))

        color = adapter.run_probe_text_only(
            text="A red bus is driving on a city street.",
            question="What color is the bus?",
        )
        valid_colors = set(adapter.config.color_vocab) | {"unknown"}
        self.assertIn(color.answer_text, valid_colors)
        self.assertTrue(color.raw_text)
        self.assertGreater(color.answer_dist.numel(), 0)
        self.assertEqual(tuple(color.features.shape), (3,))

    def test_multimodal_path_runs_on_real_prepared_example(self) -> None:
        adapter = self.adapter
        example = self.multimodal_example

        multimodal = adapter.run_backbone_multimodal(
            image=example["image_path"],
            text=example["text_input"],
            question=example["question"],
        )
        self._assert_answer_shape(multimodal.answer_text, example["family"])
        self.assertTrue(multimodal.raw_text)
        self.assertGreater(multimodal.answer_dist.numel(), 0)
        self.assertEqual(tuple(multimodal.hidden_states.shape), (adapter.config.seq_len, adapter.config.hidden_size))

        vision = adapter.run_probe_vision_only(
            image=example["image_path"],
            question=example["question"],
        )
        self._assert_answer_shape(vision.answer_text, example["family"])
        self.assertTrue(vision.raw_text)
        self.assertGreater(vision.answer_dist.numel(), 0)
        self.assertEqual(tuple(vision.features.shape), (3,))


if __name__ == "__main__":
    unittest.main()
