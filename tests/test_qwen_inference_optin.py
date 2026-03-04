from __future__ import annotations

import os
import unittest

from carm.models.backbone import BackboneConfig, Qwen25VLAdapter


@unittest.skipUnless(
    os.environ.get("RUN_QWEN_INFERENCE_TESTS") == "1",
    "set RUN_QWEN_INFERENCE_TESTS=1 to run real Qwen inference tests",
)
class TestQwenInferenceOptIn(unittest.TestCase):
    def test_text_probe_runs_real_model(self) -> None:
        adapter = Qwen25VLAdapter(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            config=BackboneConfig(hidden_size=128, seq_len=32),
            device="cpu",
            torch_dtype="float32",
            cache_results=False,
        )
        out = adapter.run_probe_text_only(
            text="A red bus is driving on a city street.",
            question="What color is the bus?",
        )
        self.assertEqual(tuple(out.answer_dist.shape), (len(adapter.config.vocab),))
        self.assertTrue(out.answer_text)
        self.assertEqual(tuple(out.features.shape), (3,))


if __name__ == "__main__":
    unittest.main()
