"""Interruptions must preserve committed output and resumable checkpoints."""

import gzip
import json
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dataset_create.instruction.client import QwenClient
from dataset_create.instruction.export import write_r2r
from dataset_create.instruction.pipeline import clear_completed_work
from dataset_create.instruction.training import export_clean_erp, validate_erp


ERP = {"width": 32, "height": 16, "jpeg_quality": 95, "jpeg_subsampling": 1}
EPISODE = {"trajectory_id": "example", "action_ids": [1, 1, 0]}
STATES = [{"position": [0., 0., z], "rotation_xyzw": [0., 0., 0., 1.]} for z in (0., -1., -2., -2.)]


def killed_export(root):
    class Renderer:
        count = 0

        def observe_erp(self, state):
            self.count += 1
            if self.count == 2:
                os.kill(os.getpid(), signal.SIGKILL)
            return np.zeros((16, 32, 3), dtype=np.uint8)
    export_clean_erp(Renderer(), EPISODE, STATES, root, ERP)


class ResumeTests(unittest.TestCase):
    def test_killed_erp_worker_leaves_no_partial_sequence_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = multiprocessing.get_context("spawn").Process(target=killed_export, args=(tmp,))
            worker.start()
            worker.join(timeout=15)
            if worker.is_alive():
                worker.kill()
                worker.join()
                self.fail("Test worker did not reach the interruption")
            self.assertEqual(worker.exitcode, -signal.SIGKILL)
            self.assertTrue((Path(tmp) / ".example.partial" / "frame_0.jpg").is_file())
            self.assertFalse((Path(tmp) / "example").exists())
            renderer = Mock()
            renderer.observe_erp.return_value = np.full((16, 32, 3), 90, dtype=np.uint8)
            export_clean_erp(renderer, EPISODE, STATES, tmp, ERP)
            validate_erp(tmp, "example", 3, ERP, inspect_images=True)
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, {"example"})

    def test_cached_model_response_is_reused_after_later_interruption(self):
        settings = json.loads((Path(__file__).parents[1] / "config/default.json").read_text())["model"]
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": '{"text":"saved"}'}}]}
        content = [{"type": "text", "text": "a navigation task"}]
        with tempfile.TemporaryDirectory() as tmp:
            client = QwenClient(settings, tmp)
            client.session.post = Mock(side_effect=[response, KeyboardInterrupt()])
            saved = client.complete("author", "system", content)
            with self.assertRaises(KeyboardInterrupt):
                client.complete("verify", "system", content)
            resumed = QwenClient(settings, tmp)
            resumed.session.post = Mock(return_value=response)
            self.assertEqual(resumed.complete("author", "system", content), saved)
            resumed.session.post.assert_not_called()
            resumed.complete("verify", "system", content)
            self.assertEqual(resumed.calls, 1)
            self.assertEqual(resumed.cache_hits, 1)

    def test_export_pair_is_rebuilt_after_interrupted_publication(self):
        original = Path.replace
        def interrupt_gzip(path, target):
            if path.name.endswith(".gz"):
                raise KeyboardInterrupt()
            return original(path, target)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json.gz"
            with patch.object(Path, "replace", interrupt_gzip):
                with self.assertRaises(KeyboardInterrupt):
                    write_r2r([], path)
            self.assertTrue(path.with_suffix("").exists())
            write_r2r([], path)
            with gzip.open(path, "rt") as stream:
                self.assertEqual(json.load(stream), json.loads(path.with_suffix("").read_text()))
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, {"train.json", "train.json.gz"})

    def test_interrupted_cleanup_keeps_completion_receipt(self):
        import shutil
        original = shutil.rmtree
        def interrupted(path, *args, **kwargs):
            original(path, *args, **kwargs)
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            (work / "episodes").mkdir(parents=True)
            (work / "manifest.json").write_text('{"export_complete":true}')
            with patch("dataset_create.instruction.pipeline.shutil.rmtree", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    clear_completed_work(work)
            self.assertTrue(json.loads((work / "manifest.json").read_text())["export_complete"])
            clear_completed_work(work)
            self.assertFalse(work.exists())


if __name__ == "__main__":
    unittest.main()
