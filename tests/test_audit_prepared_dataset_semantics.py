from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from carm.data.io import save_examples
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
    protocol_category: str,
    oracle_action: Action,
    vision_info_state: str,
    text_info_state: str,
    pairwise_relation: str,
    joint_answer: str,
    vision_supported_target: str | None,
    text_supported_target: str | None,
) -> ConflictExample:
    return ConflictExample(
        example_id=f"{protocol_category.lower()}::row",
        base_id=protocol_category.lower(),
        variant_id="row",
        image_path="debug.jpg",
        text_input=f"{protocol_category} caption",
        question="Is there a cat?",
        gold_answer="yes",
        split=Split.TRAIN,
        family=Family.EXISTENCE,
        operator=Operator.CLEAN,
        corrupt_modality=CorruptModality.NONE,
        severity=0,
        answer_type=AnswerType.BOOLEAN,
        oracle_action=oracle_action,
        evidence_modality=EvidenceModality.BOTH,
        vision_supported_target=vision_supported_target,
        text_supported_target=text_supported_target,
        vision_info_state=vision_info_state,
        text_info_state=text_info_state,
        pairwise_relation=pairwise_relation,
        joint_answer=joint_answer,
        metadata={
            "protocol_category": protocol_category,
        },
    )


class TestAuditPreparedDatasetSemantics(unittest.TestCase):
    def test_script_writes_expected_category_audit(self) -> None:
        from scripts import audit_prepared_dataset_semantics

        rows = [
            _make_example(
                protocol_category="C1",
                oracle_action=Action.REQUIRE_AGREEMENT,
                vision_info_state="informative",
                text_info_state="informative",
                pairwise_relation="consistent",
                joint_answer="yes",
                vision_supported_target="yes",
                text_supported_target="yes",
            ),
            _make_example(
                protocol_category="C2",
                oracle_action=Action.TRUST_VISION,
                vision_info_state="informative",
                text_info_state="uninformative",
                pairwise_relation="asymmetric",
                joint_answer="yes",
                vision_supported_target="yes",
                text_supported_target=None,
            ),
            _make_example(
                protocol_category="C3",
                oracle_action=Action.TRUST_TEXT,
                vision_info_state="uninformative",
                text_info_state="informative",
                pairwise_relation="asymmetric",
                joint_answer="yes",
                vision_supported_target=None,
                text_supported_target="yes",
            ),
            _make_example(
                protocol_category="C4",
                oracle_action=Action.ABSTAIN,
                vision_info_state="informative",
                text_info_state="informative",
                pairwise_relation="contradictory",
                joint_answer="<ABSTAIN>",
                vision_supported_target="yes",
                text_supported_target="no",
            ),
            _make_example(
                protocol_category="C5",
                oracle_action=Action.ABSTAIN,
                vision_info_state="uninformative",
                text_info_state="uninformative",
                pairwise_relation="both_weak",
                joint_answer="<ABSTAIN>",
                vision_supported_target=None,
                text_supported_target=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_jsonl = root / "tiny.jsonl"
            output_dir = root / "audit"
            save_examples(input_jsonl, rows)

            with patch.object(
                sys,
                "argv",
                [
                    "audit_prepared_dataset_semantics.py",
                    "--input_jsonl",
                    str(input_jsonl),
                    "--output_dir",
                    str(output_dir),
                ],
            ):
                with redirect_stdout(io.StringIO()):
                    audit_prepared_dataset_semantics.main()

            mapping = json.loads((output_dir / "category_mapping_checks.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["C1"]["joint_action_mismatches"], 0)
            self.assertEqual(mapping["C2"]["pairwise_relation_mismatches"], 0)
            self.assertEqual(mapping["C5"]["joint_target_mismatches"], 0)
            self.assertTrue((output_dir / "data_sanity_report.md").exists())
            self.assertTrue((output_dir / "category_example_sheet.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
