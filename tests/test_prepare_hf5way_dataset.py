from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from scripts import prepare_hf_5way_dataset
from scripts.prepare_hf_5way_dataset import (
    extract_contradiction_text_supported_answer,
    resolve_protocol_oracle_action,
)


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


def _c1_row(*, example_id: str = "vqa-c1::clean") -> dict[str, object]:
    return {
        "example_id": example_id,
        "question_family": "existence",
        "oracle_action": "require_agreement",
        "image_state": "clean",
        "caption_state": "clean",
        "clean_caption": "A cat sits on a chair.",
        "perturbed_caption": None,
        "question": "Is there a cat?",
        "gold_answer": "yes",
        "image_path": Image.new("RGB", (2, 2), "white"),
        "category": "CAT_1",
    }


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


def _c3_row() -> dict[str, object]:
    return {
        "example_id": "vqa-c3::clean",
        "question_family": "existence",
        "oracle_action": "trust_image",
        "image_state": "clean",
        "caption_state": "irrelevant",
        "clean_caption": "A cat sits on a chair.",
        "perturbed_caption": "A plant next to a window.",
        "question": "Is there a cat?",
        "gold_answer": "yes",
        "image_path": Image.new("RGB", (2, 2), "white"),
        "category": "CAT_2",
    }


def _c4_row() -> dict[str, object]:
    return {
        "example_id": "vqa-c4::clean",
        "question_family": "existence",
        "oracle_action": "trust_text",
        "image_state": "irrelevant",
        "caption_state": "clean",
        "clean_caption": "A cat sits on a chair.",
        "perturbed_caption": None,
        "question": "Is there a cat?",
        "gold_answer": "yes",
        "image_path": Image.new("RGB", (2, 2), "white"),
        "category": "CAT_3",
    }


def _c5_row() -> dict[str, object]:
    return {
        "example_id": "vqa-c5::clean",
        "question_family": "existence",
        "oracle_action": "abstain",
        "image_state": "irrelevant",
        "caption_state": "irrelevant",
        "clean_caption": "A cat sits on a chair.",
        "perturbed_caption": "A plant next to a window.",
        "question": "Is there a cat?",
        "gold_answer": "yes",
        "image_path": Image.new("RGB", (2, 2), "white"),
        "category": "CAT_5",
    }


class TestPrepareHF5WayDataset(unittest.TestCase):
    def test_resolve_protocol_oracle_action_rewrites_stale_c4_label(self) -> None:
        action, rewritten = resolve_protocol_oracle_action("require_agreement", "C4")
        self.assertEqual(action, "abstain")
        self.assertTrue(rewritten)

    def test_resolve_protocol_oracle_action_keeps_matching_label(self) -> None:
        action, rewritten = resolve_protocol_oracle_action("require_agreement", "C1")
        self.assertEqual(action, "require_agreement")
        self.assertFalse(rewritten)

    def test_extract_contradiction_text_supported_answer_reads_explicit_field(self) -> None:
        row = {"text_supported_target": "no"}
        value, source = extract_contradiction_text_supported_answer(
            row,
            "C4",
            question="Is there a cat?",
            family="existence",
            caption="A dog sits on a chair.",
        )
        self.assertEqual(value, "no")
        self.assertEqual(source, "explicit:text_supported_target")

    def test_extract_contradiction_text_supported_answer_derives_existence_from_caption(self) -> None:
        value, source = extract_contradiction_text_supported_answer(
            {},
            "C4",
            question="Is there a cat?",
            family="existence",
            caption="A dog sits on a chair.",
        )
        self.assertEqual(value, "no")
        self.assertEqual(source, "derived_from_caption_rule")

    def test_extract_contradiction_text_supported_answer_derives_count_from_caption(self) -> None:
        value, source = extract_contradiction_text_supported_answer(
            {},
            "C4",
            question="How many dogs are there?",
            family="count",
            caption="Three dogs stand near a fence.",
        )
        self.assertEqual(value, "3")
        self.assertEqual(source, "derived_from_caption_rule")

    def test_extract_contradiction_text_supported_answer_derives_color_from_caption(self) -> None:
        value, source = extract_contradiction_text_supported_answer(
            {},
            "C4",
            question="What color is the shirt?",
            family="attribute_color",
            caption="A man wearing a blue shirt walks outside.",
        )
        self.assertEqual(value, "blue")
        self.assertEqual(source, "derived_from_caption_rule")

    def test_extract_contradiction_text_supported_answer_marks_missing_when_caption_rule_fails(self) -> None:
        value, source = extract_contradiction_text_supported_answer(
            {},
            "C4",
            question="What color is the shirt?",
            family="attribute_color",
            caption="A man walks outside.",
        )
        self.assertIsNone(value)
        self.assertEqual(source, "missing_after_caption_rule")

    def test_non_c4_rows_do_not_require_text_supported_answer(self) -> None:
        self.assertEqual(
            extract_contradiction_text_supported_answer(
                {},
                "C1",
                question="Is there a cat?",
                family="existence",
                caption="A cat sits on a chair.",
            ),
            (None, None),
        )

    def test_main_writes_explicit_c4_targets(self) -> None:
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
        self.assertEqual(row["vision_info_state"], "informative")
        self.assertEqual(row["text_info_state"], "informative")
        self.assertEqual(row["pairwise_relation"], "contradictory")
        self.assertEqual(row["joint_answer"], "<ABSTAIN>")
        self.assertNotIn("c2_text_supported_answer", row["metadata"])
        self.assertEqual(row["metadata"]["raw_oracle_action"], "require_agreement")
        self.assertEqual(row["metadata"]["text_supported_target_source"], "explicit:text_supported_target")
        self.assertEqual(row["metadata"]["vision_target_source"], "gold_answer")
        self.assertEqual(row["metadata"]["text_target_source"], "explicit:text_supported_target")
        self.assertEqual(row["metadata"]["target_derivation_status"], "ok")
        self.assertTrue(row["metadata"]["contradiction_supervision_available"])
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["oracle_action_rewrite_count"], 1)
        self.assertEqual(manifest["contradiction_target_counts"]["contradiction_rows"], 1)
        self.assertEqual(manifest["contradiction_target_counts"]["contradiction_validated"], 1)
        self.assertEqual(manifest["contradiction_target_counts"]["vision_supported_target"], 1)
        self.assertEqual(manifest["contradiction_target_counts"]["text_supported_target"], 1)
        self.assertEqual(manifest["contradiction_text_target_source_counts"]["explicit:text_supported_target"], 1)
        self.assertEqual(manifest["validated_contradiction_coverage"], 1.0)

    def test_main_derives_c4_target_when_explicit_field_is_missing(self) -> None:
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
        self.assertEqual(row["metadata"]["target_derivation_status"], "ok")
        self.assertTrue(row["metadata"]["contradiction_supervision_available"])
        self.assertEqual(manifest["contradiction_text_target_source_counts"]["derived_from_caption_rule"], 1)
        self.assertEqual(manifest["contradiction_target_counts"]["contradiction_validated"], 1)

    def test_main_records_missing_c4_target_when_caption_rule_cannot_derive(self) -> None:
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

        self.assertEqual(manifest["status"], "ok_partial_contradiction_targets")
        self.assertEqual(manifest["contradiction_target_counts"]["text_supported_target_missing"], 1)
        self.assertEqual(manifest["contradiction_text_target_source_counts"]["missing_after_caption_rule"], 1)
        self.assertEqual(manifest["contradiction_text_target_coverage"], 0.0)
        self.assertEqual(manifest["validated_contradiction_coverage"], 0.0)
        self.assertEqual(manifest["missing_contradiction_text_supported_target_examples_preview"], ["vqa-1::clean"])
        self.assertIsNone(row["text_supported_target"])
        self.assertEqual(row["metadata"]["text_supported_target_source"], "missing_after_caption_rule")
        self.assertEqual(row["metadata"]["target_derivation_status"], "partial")
        self.assertEqual(row["metadata"]["target_mask_reason"], "missing_text_supported_target")
        self.assertFalse(row["metadata"]["contradiction_supervision_available"])

    def test_main_marks_noncontradictory_c4_when_targets_agree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = _base_args(root)
            with (
                patch.object(prepare_hf_5way_dataset, "parse_args", return_value=args),
                patch.object(prepare_hf_5way_dataset, "_load_hf_rows", return_value=[_c2_row(text_supported_target="yes")]),
                patch.object(prepare_hf_5way_dataset, "_resolve_sha", return_value="sha-unit"),
            ):
                prepare_hf_5way_dataset.main()

            manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
            row = json.loads(Path(args.output_jsonl).read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(manifest["status"], "ok_partial_contradiction_targets")
        self.assertEqual(manifest["contradiction_target_counts"]["contradiction_not_validated_targets_agree"], 1)
        self.assertEqual(manifest["noncontradictory_examples_preview"], ["vqa-1::clean"])
        self.assertEqual(row["text_supported_target"], "yes")
        self.assertEqual(row["metadata"]["target_derivation_status"], "partial")
        self.assertEqual(row["metadata"]["target_mask_reason"], "targets_agree_after_canonicalization")
        self.assertFalse(row["metadata"]["contradiction_supervision_available"])

    def test_main_writes_category_level_supervision_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = _base_args(root)
            rows = [_c1_row(), _c3_row(), _c4_row(), _c5_row()]
            with (
                patch.object(prepare_hf_5way_dataset, "parse_args", return_value=args),
                patch.object(prepare_hf_5way_dataset, "_load_hf_rows", return_value=rows),
                patch.object(prepare_hf_5way_dataset, "_resolve_sha", return_value="sha-unit"),
            ):
                prepare_hf_5way_dataset.main()

            written_rows = [
                json.loads(line)
                for line in Path(args.output_jsonl).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        by_id = {row["example_id"]: row for row in written_rows}
        self.assertEqual(by_id["vqa-c1::clean"]["vision_supported_target"], "yes")
        self.assertEqual(by_id["vqa-c1::clean"]["text_supported_target"], "yes")
        self.assertEqual(by_id["vqa-c1::clean"]["pairwise_relation"], "consistent")
        self.assertEqual(by_id["vqa-c1::clean"]["joint_answer"], "yes")

        self.assertEqual(by_id["vqa-c3::clean"]["vision_supported_target"], "yes")
        self.assertIsNone(by_id["vqa-c3::clean"]["text_supported_target"])
        self.assertEqual(by_id["vqa-c3::clean"]["pairwise_relation"], "asymmetric")
        self.assertEqual(by_id["vqa-c3::clean"]["joint_answer"], "yes")

        self.assertIsNone(by_id["vqa-c4::clean"]["vision_supported_target"])
        self.assertEqual(by_id["vqa-c4::clean"]["text_supported_target"], "yes")
        self.assertEqual(by_id["vqa-c4::clean"]["pairwise_relation"], "asymmetric")
        self.assertEqual(by_id["vqa-c4::clean"]["joint_answer"], "yes")

        self.assertIsNone(by_id["vqa-c5::clean"]["vision_supported_target"])
        self.assertIsNone(by_id["vqa-c5::clean"]["text_supported_target"])
        self.assertEqual(by_id["vqa-c5::clean"]["pairwise_relation"], "both_weak")
        self.assertEqual(by_id["vqa-c5::clean"]["joint_answer"], "<ABSTAIN>")

    def test_main_uses_official_hf_splits_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = _base_args(root)
            args.hf_split = "all"
            dataset_dict = {
                "train": [_c1_row(example_id="vqa-train::clean")],
                "validation": [_c1_row(example_id="vqa-val::clean")],
                "test": [_c1_row(example_id="vqa-test::clean")],
            }
            with (
                patch.object(prepare_hf_5way_dataset, "parse_args", return_value=args),
                patch.object(prepare_hf_5way_dataset, "_load_hf_rows", return_value=dataset_dict),
                patch.object(prepare_hf_5way_dataset, "_resolve_sha", return_value="sha-unit"),
            ):
                prepare_hf_5way_dataset.main()

            rows = [
                json.loads(line)
                for line in Path(args.output_jsonl).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))

        split_by_example = {row["example_id"]: row["split"] for row in rows}
        self.assertEqual(split_by_example["vqa-train::clean"], "train")
        self.assertEqual(split_by_example["vqa-val::clean"], "val")
        self.assertEqual(split_by_example["vqa-test::clean"], "test_id")
        self.assertEqual(manifest["split_assignment_mode"], "hf_official")
        self.assertFalse(manifest["ratios_applied"])
        self.assertEqual(manifest["hf_source_split_available_counts"], {"test": 1, "train": 1, "validation": 1})
        self.assertEqual(manifest["hf_source_split_read_counts"], {"test": 1, "train": 1, "validation": 1})
        self.assertEqual(manifest["split_counts"], {"test_id": 1, "train": 1, "val": 1})


if __name__ == "__main__":
    unittest.main()
