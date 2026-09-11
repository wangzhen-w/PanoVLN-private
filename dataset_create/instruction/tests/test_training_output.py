"""Training-frame alignment, atomic publication and final artifact contracts."""

import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from dataset_create.instruction.export import write_r2r
from dataset_create.instruction.pipeline import clear_episode_materials
from dataset_create.instruction.training import export_clean_erp, training_states, validate_erp


class TrainingOutputTests(unittest.TestCase):
    def setUp(self):
        self.episode = {"trajectory_id": "example", "action_ids": [1, 2, 0]}
        self.states = [
            {"position": [0., 0., 0.], "rotation_xyzw": [0., 0., 0., 1.]},
            {"position": [0., 0., -1.], "rotation_xyzw": [0., 0., 0., 1.]},
            {"position": [0., 0., -1.], "rotation_xyzw": [0., .1305262, 0., .9914449]},
        ]
        self.states.append(copy.deepcopy(self.states[-1]))
        self.settings = {"width": 32, "height": 16, "jpeg_quality": 95, "jpeg_subsampling": 1}

    def test_every_action_has_its_pre_action_observation(self):
        frames = training_states(self.episode, self.states)
        self.assertEqual(frames, self.states[:3])
        self.assertEqual(len(frames), len(self.episode["action_ids"]))
        self.assertNotEqual(frames[1]["rotation_xyzw"], frames[2]["rotation_xyzw"])

    def test_nonterminal_or_moving_stop_is_rejected(self):
        bad = {**self.episode, "action_ids": [0, 1, 2]}
        with self.assertRaises(ValueError):
            training_states(bad, self.states)
        self.states[-1]["position"][0] = 1.
        with self.assertRaises(ValueError):
            training_states(self.episode, self.states)

    def test_clean_erp_publication_and_resume(self):
        renderer = Mock()
        renderer.observe_erp.return_value = np.full((16, 32, 3), 120, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            export_clean_erp(renderer, self.episode, self.states, tmp, self.settings)
            self.assertEqual(renderer.observe_erp.call_count, 3)
            self.assertEqual(renderer.observe_erp.call_args_list[-1].args[0], self.states[-2])
            validate_erp(tmp, "example", 3, self.settings, inspect_images=True)
            export_clean_erp(renderer, self.episode, self.states, tmp, self.settings)
            self.assertEqual(renderer.observe_erp.call_count, 3)
            self.assertEqual({p.name for p in (Path(tmp) / "example").iterdir()},
                             {"frame_0.jpg", "frame_1.jpg", "frame_2.jpg"})

    def test_failed_render_does_not_publish_partial_erp(self):
        renderer = Mock()
        renderer.observe_erp.side_effect = [np.zeros((16, 32, 3), dtype=np.uint8), RuntimeError("render failed")]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                export_clean_erp(renderer, self.episode, self.states, tmp, self.settings)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_both_r2r_serializations_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "train.json.gz"
            write_r2r([], target)
            with gzip.open(target, "rt") as stream:
                compressed = json.load(stream)
            self.assertEqual(json.loads(target.with_suffix("").read_text()), compressed)
            self.assertEqual(set(compressed), {"episodes", "instruction_vocab"})

    def test_material_cleanup_preserves_record_and_training_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "episode"
            for name in ("media", "requests"):
                (root / name).mkdir(parents=True)
                (root / name / "temporary").write_text("temporary")
            (root / "ground_route.npz").write_bytes(b"temporary")
            (root / "record.json").write_text("{}")
            erp = Path(tmp) / "erp"
            erp.mkdir()
            (erp / "frame_0.jpg").write_bytes(b"keep")
            clear_episode_materials(root)
            self.assertEqual([p.name for p in root.iterdir()], ["record.json"])
            self.assertEqual((erp / "frame_0.jpg").read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
