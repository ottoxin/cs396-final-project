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
        vision_supported_target="yes" if protocol_category == "C4" else None,
        text_supported_target="no" if protocol_category == "C4" else None,
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


class TestExperimentalSmallRun(unittest.TestCase):
    def test_script_writes_required_artifacts(self) -> None:
        from scripts import run_experimental_small_data

        rows = _tiny_dataset()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_jsonl = root / "tiny.jsonl"
            output_dir = root / "outputs"
            config_path = root / "config.yaml"
            save_examples(input_jsonl, rows)
            config_path.write_text(
                "\n".join(
                    [
                        "seed: 7",
                        "backbone:",
                        "  name: deterministic_debug_backbone",
                        "model:",
                        "  hidden_size: 128",
                        "  probe_feature_size: 3",
                        "  pool: mean",
                        "training:",
                        "  device: auto",
                        "eval:",
                        "  confidence_threshold: 0.3",
                        "experimental:",
                        "  run_name: TEST-EXP",
                        "  random_seed: 7",
                        "  sampling_strategy: protocol_category_family",
                        "  max_train_examples: 5",
                        "  max_val_examples: 5",
                        "  max_test_examples: 5",
                        "  training:",
                        "    batch_size: 2",
                        "    epochs: 2",
                        "    patience: 1",
                        "    log_every_steps: 1",
                        "  loss:",
                        "    lambda_vision_info: 1.0",
                        "    lambda_text_info: 1.0",
                        "    lambda_relation: 1.0",
                        "    lambda_action: 1.0",
                        "    lambda_cf: 0.0",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_experimental_small_data.py",
                    "--config",
                    str(config_path),
                    "--input_jsonl",
                    str(input_jsonl),
                    "--output_dir",
                    str(output_dir),
                ],
            ):
                run_stdout = io.StringIO()
                with redirect_stdout(run_stdout):
                    run_experimental_small_data.main()

            expected = [
                output_dir / "run_config.json",
                output_dir / "split_manifest.json",
                output_dir / "split_stats.json",
                output_dir / "label_derivation_audit.csv",
                output_dir / "label_derivation_audit.jsonl",
                output_dir / "data_schema_audit_summary.md",
                output_dir / "failure_report.md",
                output_dir / "baseline_comparison.csv",
                output_dir / "baseline_comparison.md",
                output_dir / "prompt_templates" / "prompt_only_abstain_action.txt",
                output_dir / "structured_carm" / "train" / "structured_carm_heads.pt",
                output_dir / "structured_carm" / "train" / "train_progress.jsonl",
                output_dir / "structured_carm" / "test" / "metrics.json",
                output_dir / "structured_carm" / "test" / "per_example_predictions.jsonl",
                output_dir / "structured_carm" / "test" / "failure_diagnostics.csv",
                output_dir / "structured_carm" / "test" / "feature_diagnostics_summary.md",
            ]
            for path in expected:
                self.assertTrue(path.exists(), path)
            self.assertIn('"phase": "train_progress"', run_stdout.getvalue())
            self.assertIn('"avg_loss_total"', run_stdout.getvalue())

            metrics = json.loads((output_dir / "structured_carm" / "test" / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("task_success_revised", metrics)
            self.assertIn("action_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
