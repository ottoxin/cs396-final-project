from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from scripts import prepare_hf_5way_dataset
from scripts.prepare_hf_5way_dataset import extract_c2_text_supported_answer, resolve_protocol_oracle_action


def _base_args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        hf_repo_id="nbso/carm-vqa-5way",
        hf_revision="unit-test",
        hf_split="train",
        cache_root=str(root / "cache"),
        output_jsonl=str(root / "prepared.jsonl"),
        manifest_json=str(root / "manifest.json"),
        image_dir=str(root / "images"),
        seed=7,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        max_rows=None,
        jpeg_quality=90,
    )


def _c2_row(*, text_supported_target: str | None = "no") -> dict[str, object]:
    row: dict[str, object] = {
        "example_id": "vqa-1::clean",
        "question_family": "existence",
        "oracle_action": "require_agreement",
        "image_state": "clean",
        "caption_state": "different",
        "clean_caption": "A cat sits on a chair.",
        "perturbed_caption": "A dog sits on a chair.",
        "question": "Is there a cat?",
        "gold_answer": "yes",
        "image_path": Image.new("RGB", (2, 2), "white"),
        "category": "CAT_4",
    }
    if text_supported_target is not None:
        row["text_supported_target"] = text_supported_target
    return row


class TestPrepareHF5WayDataset(unittest.TestCase):
    def test_resolve_protocol_oracle_action_rewrites_stale_c2_label(self) -> None:
        action, rewritten = resolve_protocol_oracle_action("require_agreement", "C2")
        self.assertEqual(action, "abstain")
        self.assertTrue(rewritten)

    def test_resolve_protocol_oracle_action_keeps_matching_label(self) -> None:
        action, rewritten = resolve_protocol_oracle_action("require_agreement", "C1")
        self.assertEqual(action, "require_agreement")
        self.assertFalse(rewritten)

    def test_extract_c2_text_supported_answer_reads_explicit_field(self) -> None:
        row = {"text_supported_target": "no"}
        self.assertEqual(extract_c2_text_supported_answer(row, "C2"), "no")

    def test_extract_c2_text_supported_answer_rejects_metadata_proxy(self) -> None:
        row = {"metadata": {"caption_supported_answer": "purple"}}
        with self.assertRaises(ValueError):
            extract_c2_text_supported_answer(row, "C2")

    def test_extract_c2_text_supported_answer_raises_when_missing(self) -> None:
        with self.assertRaises(ValueError):
            extract_c2_text_supported_answer({}, "C2")

    def test_non_c2_rows_do_not_require_text_supported_answer(self) -> None:
        self.assertIsNone(extract_c2_text_supported_answer({}, "C1"))

    def test_main_writes_explicit_c2_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = _base_args(root)
            with (
                patch.object(prepare_hf_5way_dataset, "parse_args", return_value=args),
                patch.object(prepare_hf_5way_dataset, "_load_hf_rows", return_value=[_c2_row()]),
                patch.object(prepare_hf_5way_dataset, "_resolve_sha", return_value="sha-unit"),
            ):
                prepare_hf_5way_dataset.main()

            output_path = Path(args.output_jsonl)
            manifest_path = Path(args.manifest_json)
            row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(row["oracle_action"], "abstain")
        self.assertEqual(row["vision_supported_target"], "yes")
        self.assertEqual(row["text_supported_target"], "no")
        self.assertNotIn("c2_text_supported_answer", row["metadata"])
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["oracle_action_rewrite_count"], 1)
        self.assertEqual(manifest["c2_target_counts"]["c2_rows"], 1)
        self.assertEqual(manifest["c2_target_counts"]["vision_supported_target"], 1)
        self.assertEqual(manifest["c2_target_counts"]["text_supported_target"], 1)

    def test_main_fails_loudly_when_explicit_c2_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = _base_args(root)
            with (
                patch.object(prepare_hf_5way_dataset, "parse_args", return_value=args),
                patch.object(prepare_hf_5way_dataset, "_load_hf_rows", return_value=[_c2_row(text_supported_target=None)]),
                patch.object(prepare_hf_5way_dataset, "_resolve_sha", return_value="sha-unit"),
            ):
                with self.assertRaises(SystemExit):
                    prepare_hf_5way_dataset.main()

            manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure_reason"], "missing_c2_text_supported_target")
        self.assertEqual(manifest["drop_counts"]["missing_c2_text_supported_target"], 1)
        self.assertEqual(manifest["missing_c2_text_supported_target_examples_preview"], ["vqa-1::clean"])
        self.assertFalse(Path(args.output_jsonl).exists())


if __name__ == "__main__":
    unittest.main()
