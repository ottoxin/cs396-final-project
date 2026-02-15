from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from carm.data.io import save_examples
from carm.data.schema import Split
from carm.eval.evaluator import CARMPredictor, evaluate_predictor
from carm.models.backbone import MockFrozenBackbone
from carm.models.carm_model import CARMHeads
from carm.train.trainer import CARMTrainer, TrainerConfig
from tests.fixtures import make_examples


class TestIntegrationTrainEval(unittest.TestCase):
    def test_single_batch_train_and_eval_artifacts(self) -> None:
        examples = make_examples()
        train_examples = [ex for ex in examples if ex.split == Split.TRAIN]
        test_examples = [ex for ex in examples if ex.split == Split.TEST]

        model = CARMHeads()
        backbone = MockFrozenBackbone()
        trainer = CARMTrainer(model=model, backbone=backbone, config=TrainerConfig(batch_size=2, epochs=1, device="cpu"))

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            train_dir = out / "train"
            eval_dir = out / "eval"

            metrics = trainer.train(train_examples, output_dir=train_dir)
            self.assertIn("loss_total", metrics)
            self.assertTrue((train_dir / "train_metrics.json").exists())

            predictor = CARMPredictor(model=model, backbone=backbone)
            eval_metrics = evaluate_predictor(predictor, test_examples, output_dir=eval_dir)
            self.assertIn("accuracy", eval_metrics)

            pred_file = eval_dir / "per_example_predictions.jsonl"
            self.assertTrue(pred_file.exists())
            line = pred_file.read_text(encoding="utf-8").splitlines()[0]
            row = json.loads(line)

            required = {
                "pred_conflict_type",
                "pred_action",
                "r_v",
                "r_t",
                "abstained",
                "final_answer",
                "correct",
            }
            self.assertTrue(required.issubset(set(row.keys())))

            snapshot = train_dir / "train_examples_snapshot.jsonl"
            self.assertTrue(snapshot.exists())

            loaded = snapshot.read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(loaded), 0)


if __name__ == "__main__":
    unittest.main()
