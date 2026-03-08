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
        value, source = extract_c2_text_supported_answer(
            row,
            "C2",
            question="Is there a cat?",
            family="existence",
            caption="A dog sits on a chair.",
        )
        self.assertEqual(value, "no")
        self.assertEqual(source, "explicit:text_supported_target")

    def test_extract_c2_text_supported_answer_derives_existence_from_caption(self) -> None:
        value, source = extract_c2_text_supported_answer(
            {},
            "C2",
            question="Is there a cat?",
            family="existence",
            caption="A dog sits on a chair.",
        )
        self.assertEqual(value, "no")
        self.assertEqual(source, "derived_from_caption_rule")

    def test_extract_c2_text_supported_answer_derives_count_from_caption(self) -> None:
        value, source = extract_c2_text_supported_answer(
            {},
            "C2",
            question="How many dogs are there?",
            family="count",
            caption="Three dogs stand near a fence.",
        )
        self.assertEqual(value, "3")
        self.assertEqual(source, "derived_from_caption_rule")

    def test_extract_c2_text_supported_answer_derives_color_from_caption(self) -> None:
        value, source = extract_c2_text_supported_answer(
            {},
            "C2",
            question="What color is the shirt?",
            family="attribute_color",
            caption="A man wearing a blue shirt walks outside.",
        )
        self.assertEqual(value, "blue")
        self.assertEqual(source, "derived_from_caption_rule")

    def test_extract_c2_text_supported_answer_marks_missing_when_caption_rule_fails(self) -> None:
        value, source = extract_c2_text_supported_answer(
            {},
            "C2",
            question="What color is the shirt?",
            family="attribute_color",
            caption="A man walks outside.",
        )
        self.assertIsNone(value)
        self.assertEqual(source, "missing_after_caption_rule")

    def test_non_c2_rows_do_not_require_text_supported_answer(self) -> None:
        self.assertEqual(
            extract_c2_text_supported_answer(
                {},
                "C1",
                question="Is there a cat?",
                family="existence",
                caption="A cat sits on a chair.",
            ),
            (None, None),
        )

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
        self.assertEqual(row["metadata"]["text_supported_target_source"], "explicit:text_supported_target")
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["oracle_action_rewrite_count"], 1)
        self.assertEqual(manifest["c2_target_counts"]["c2_rows"], 1)
        self.assertEqual(manifest["c2_target_counts"]["vision_supported_target"], 1)
        self.assertEqual(manifest["c2_target_counts"]["text_supported_target"], 1)
        self.assertEqual(manifest["c2_text_target_source_counts"]["explicit:text_supported_target"], 1)

    def test_main_derives_c2_target_when_explicit_field_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = _base_args(root)
            with (
                patch.object(prepare_hf_5way_dataset, "parse_args", return_value=args),
                patch.object(prepare_hf_5way_dataset, "_load_hf_rows", return_value=[_c2_row(text_supported_target=None)]),
                patch.object(prepare_hf_5way_dataset, "_resolve_sha", return_value="sha-unit"),
            ):
                prepare_hf_5way_dataset.main()

            manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
            row = json.loads(Path(args.output_jsonl).read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(row["text_supported_target"], "no")
        self.assertEqual(row["metadata"]["text_supported_target_source"], "derived_from_caption_rule")
        self.assertEqual(manifest["c2_text_target_source_counts"]["derived_from_caption_rule"], 1)

    def test_main_records_missing_c2_target_when_caption_rule_cannot_derive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = _base_args(root)
            bad_row = _c2_row(text_supported_target=None) | {"question_family": "attribute_color", "question": "What color is the shirt?", "perturbed_caption": "A man walks outside."}
            with (
                patch.object(prepare_hf_5way_dataset, "parse_args", return_value=args),
                patch.object(prepare_hf_5way_dataset, "_load_hf_rows", return_value=[bad_row]),
                patch.object(prepare_hf_5way_dataset, "_resolve_sha", return_value="sha-unit"),
            ):
                prepare_hf_5way_dataset.main()

            manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
            row = json.loads(Path(args.output_jsonl).read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(manifest["status"], "ok_partial_c2_text_targets")
        self.assertEqual(manifest["c2_target_counts"]["text_supported_target_missing"], 1)
        self.assertEqual(manifest["c2_text_target_source_counts"]["missing_after_caption_rule"], 1)
        self.assertEqual(manifest["c2_text_target_coverage"], 0.0)
        self.assertEqual(manifest["missing_c2_text_supported_target_examples_preview"], ["vqa-1::clean"])
        self.assertIsNone(row["text_supported_target"])
        self.assertEqual(row["metadata"]["text_supported_target_source"], "missing_after_caption_rule")


if __name__ == "__main__":
    unittest.main()
