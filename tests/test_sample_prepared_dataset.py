from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from carm.data.io import load_examples, save_examples
from carm.data.schema import (
    Action,
    AnswerType,
    ConflictExample,
    CorruptModality,
    EvidenceModality,
    Family,
    Operator,
    Split,
)


def _make_example(
    *,
    split: Split,
    idx: int,
    protocol_category: str,
    image_state: str,
    caption_state: str,
    oracle_action: Action,
    corrupt_modality: CorruptModality,
) -> ConflictExample:
    return ConflictExample(
        example_id=f"{split.value}-{protocol_category.lower()}-{idx}::row",
        base_id=f"{split.value}-{protocol_category.lower()}-{idx}",
        variant_id="row",
        image_path=f"{split.value}-{idx}.jpg",
        text_input=f"{protocol_category} caption {idx}",
        question="Is there a cat?",
        gold_answer="yes",
        split=split,
        family=Family.EXISTENCE,
        operator=Operator.CLEAN,
        corrupt_modality=corrupt_modality,
        severity=0,
        answer_type=AnswerType.BOOLEAN,
        oracle_action=oracle_action,
        evidence_modality=EvidenceModality.BOTH,
        vision_supported_target="yes" if protocol_category in {"C1", "C2", "C4"} else None,
        text_supported_target="no" if protocol_category == "C4" else ("yes" if protocol_category in {"C1", "C3"} else None),
        vision_info_state="informative" if protocol_category in {"C1", "C2", "C4"} else "uninformative",
        text_info_state="informative" if protocol_category in {"C1", "C3", "C4"} else "uninformative",
        pairwise_relation={
            "C1": "consistent",
            "C2": "asymmetric",
            "C3": "asymmetric",
            "C4": "contradictory",
            "C5": "both_weak",
        }[protocol_category],
        joint_answer="yes" if protocol_category in {"C1", "C2", "C3"} else "<ABSTAIN>",
        metadata={
            "protocol_category": protocol_category,
            "image_state": image_state,
            "caption_state": caption_state,
        },
    )


def _tiny_dataset() -> list[ConflictExample]:
    specs = [
        ("C1", "clean", "clean", Action.REQUIRE_AGREEMENT, CorruptModality.NONE),
        ("C2", "clean", "irrelevant", Action.TRUST_VISION, CorruptModality.TEXT),
        ("C3", "irrelevant", "clean", Action.TRUST_TEXT, CorruptModality.VISION),
        ("C4", "clean", "different", Action.ABSTAIN, CorruptModality.TEXT),
        ("C5", "irrelevant", "irrelevant", Action.ABSTAIN, CorruptModality.BOTH),
    ]
    rows: list[ConflictExample] = []
    for split in (Split.TRAIN, Split.VAL, Split.TEST_ID):
        for idx, spec in enumerate(specs, start=1):
            rows.append(
                _make_example(
                    split=split,
                    idx=idx,
                    protocol_category=spec[0],
                    image_state=spec[1],
                    caption_state=spec[2],
                    oracle_action=spec[3],
                    corrupt_modality=spec[4],
                )
            )
    return rows


class TestSamplePreparedDataset(unittest.TestCase):
    def test_script_writes_split_preserving_fractional_subset(self) -> None:
        from scripts import sample_prepared_dataset

        rows = _tiny_dataset()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_jsonl = root / "tiny.jsonl"
            output_jsonl = root / "tiny_40pct.jsonl"
            manifest_json = root / "tiny_40pct.manifest.json"
            save_examples(input_jsonl, rows)

            with patch.object(
                sys,
                "argv",
                [
                    "sample_prepared_dataset.py",
                    "--input_jsonl",
                    str(input_jsonl),
                    "--output_jsonl",
                    str(output_jsonl),
                    "--manifest_json",
                    str(manifest_json),
                    "--fraction",
                    "0.4",
                    "--seed",
                    "7",
                ],
            ):
                with redirect_stdout(io.StringIO()):
                    sample_prepared_dataset.main()

            sampled = load_examples(output_jsonl)
            counts = Counter(example.split.value for example in sampled)
            self.assertEqual(dict(sorted(counts.items())), {"test_id": 2, "train": 2, "val": 2})

            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_split_sizes"], {"train": 5, "val": 5, "test_id": 5})
            self.assertEqual(manifest["selected_split_sizes"], {"train": 2, "val": 2, "test_id": 2})
            self.assertEqual(manifest["total_selected_examples"], 6)


if __name__ == "__main__":
    unittest.main()
