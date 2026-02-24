from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from carm.data.schema import Family
from carm.data.vqa_coco import build_base_examples


class TestVQACocoIngestion(unittest.TestCase):
    def test_build_base_examples(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            q_train = root / "q_train.json"
            q_val = root / "q_val.json"
            a_train = root / "a_train.json"
            a_val = root / "a_val.json"
            caps = root / "caps.json"

            q_train.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "question_id": 1,
                                "image_id": 101,
                                "question": "What color is the car?",
                            },
                            {
                                "question_id": 2,
                                "image_id": 102,
                                "question": "How many dogs are there?",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            q_val.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "question_id": 3,
                                "image_id": 103,
                                "question": "Is there a bicycle in the image?",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            a_train.write_text(
                json.dumps(
                    {
                        "annotations": [
                            {"question_id": 1, "multiple_choice_answer": "red"},
                            {"question_id": 2, "multiple_choice_answer": "2"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            a_val.write_text(
                json.dumps({"annotations": [{"question_id": 3, "multiple_choice_answer": "yes"}]}),
                encoding="utf-8",
            )
            caps.write_text(
                json.dumps(
                    {
                        "annotations": [
                            {"image_id": 101, "caption": "A red car near a curb."},
                            {"image_id": 102, "caption": "Two dogs playing in grass."},
                            {"image_id": 103, "caption": "A bicycle beside a tree."},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            examples, stats = build_base_examples(
                vqa_questions_train=str(q_train),
                vqa_questions_val=str(q_val),
                vqa_annotations_train=str(a_train),
                vqa_annotations_val=str(a_val),
                coco_captions_train=str(caps),
                coco_captions_val=str(caps),
                image_train_dir="images/train2014",
                image_val_dir="images/val2014",
                families=[Family.EXISTENCE, Family.COUNT, Family.ATTRIBUTE_COLOR],
                color_vocab=["red", "blue", "green"],
                consistency_filter=True,
                seed=7,
                max_per_family=None,
            )

            self.assertEqual(len(examples), 3)
            self.assertEqual(stats["family_counts"]["attribute_color"], 1)
            self.assertEqual(stats["family_counts"]["count"], 1)
            self.assertEqual(stats["family_counts"]["existence"], 1)


if __name__ == "__main__":
    unittest.main()
