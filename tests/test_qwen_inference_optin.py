from __future__ import annotations

import os
import unittest

import torch

from carm.models.backbone import BackboneConfig, Qwen25VLAdapter


@unittest.skipUnless(
    os.environ.get("RUN_QWEN_INFERENCE_TESTS") == "1",
    "set RUN_QWEN_INFERENCE_TESTS=1 to run real Qwen inference tests",
)
class TestQwenInferenceOptIn(unittest.TestCase):
    def test_text_probe_runs_real_model_with_family_specific_parsing(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        adapter = Qwen25VLAdapter(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            config=BackboneConfig(hidden_size=128, seq_len=32),
            device=device,
            torch_dtype=dtype,
            cache_results=False,
        )

        existence = adapter.run_probe_text_only(
            text="A bicycle is next to a tree.",
            question="Is there a bicycle in the image?",
        )
        self.assertIn(existence.answer_text, {"yes", "no", "unknown"})
        self.assertGreater(existence.answer_dist.numel(), 0)
        self.assertEqual(tuple(existence.features.shape), (3,))

        count = adapter.run_probe_text_only(
            text="Two dogs sit on grass.",
            question="How many dogs are there?",
        )
        self.assertRegex(count.answer_text, r"^(unknown|\d+)$")
        self.assertGreater(count.answer_dist.numel(), 0)
        self.assertEqual(tuple(count.features.shape), (3,))

        color = adapter.run_probe_text_only(
            text="A red bus is driving on a city street.",
            question="What color is the bus?",
        )
        valid_colors = set(adapter.config.color_vocab) | {"unknown"}
        self.assertIn(color.answer_text, valid_colors)
        self.assertGreater(color.answer_dist.numel(), 0)
        self.assertEqual(tuple(color.features.shape), (3,))


if __name__ == "__main__":
    unittest.main()
