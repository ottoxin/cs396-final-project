from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_baselines import _prune_summary, _resolve_example_image_paths
from tests.fixtures import make_base_examples


class TestRunBaselines(unittest.TestCase):
    def test_prune_summary_removes_stale_baseline_keys(self) -> None:
        summary = {
            "backbone_direct": {"accuracy": 0.1},
            "agreement_check": {"accuracy": 0.2},
            "probe_heuristic": {"accuracy": 0.3},
        }
        active = {"backbone_direct", "probe_heuristic"}

        pruned, stale = _prune_summary(summary, active)

        self.assertEqual(stale, ["agreement_check"])
        self.assertEqual(set(pruned.keys()), active)

    def test_resolve_example_image_paths_normalizes_to_absolute(self) -> None:
        examples = make_base_examples()[:1]
        ex = examples[0]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / ex.image_path
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fake")

            logs: list[str] = []
            _resolve_example_image_paths(examples, logs.append, project_root=root)

            self.assertTrue(Path(ex.image_path).is_absolute())
            self.assertTrue(Path(ex.image_path).exists())
            self.assertTrue(any("validated image paths" in line for line in logs))

    def test_resolve_example_image_paths_raises_for_missing_files(self) -> None:
        examples = make_base_examples()[:1]
        ex = examples[0]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with self.assertRaises(FileNotFoundError) as ctx:
                _resolve_example_image_paths(examples, lambda _: None, project_root=root)

        msg = str(ctx.exception)
        self.assertIn("Missing image_path", msg)
        self.assertIn(ex.example_id, msg)


if __name__ == "__main__":
    unittest.main()
