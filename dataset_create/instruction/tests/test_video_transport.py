"""Transport regression: client-selected navigation frames must survive decoding.

The native decoder test also runs under a vLLM environment; it is skipped in the
Habitat-only environment. Neither test loads model weights or calls a live API.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dataset_create.instruction.client import QwenClient, media_item


class VideoTransportTests(unittest.TestCase):
    def request_parameters(self):
        settings = json.loads((Path(__file__).parents[1] / "config/default.json").read_text())["model"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes(b"transport fixture; decoding is tested independently")
            client = QwenClient(settings, Path(directory) / "requests")
            client.session.post = Mock(return_value=SimpleNamespace(status_code=200, ok=True, json=lambda: {
                "choices": [{"finish_reason": "stop", "message": {"content": '{}'}}], "usage": {}}))
            client.complete("test", "Describe the route.", [media_item(path, "video")])
            parameters = client.session.post.call_args.kwargs["json"]
            client.session.close()
            return parameters

    def test_presampled_video_request_disables_both_resampling_stages(self):
        parameters = self.request_parameters()
        self.assertEqual(parameters['media_io_kwargs']['video'],
                         {"video_backend": "opencv", "num_frames": -1, "fps": -1})
        self.assertFalse(parameters['mm_processor_kwargs']['do_sample_frames'])
        self.assertEqual(parameters['messages'][1]['content'][0]['type'], 'video_url')

    def test_native_decoder_keeps_middle_of_route(self):
        try:
            from vllm.multimodal.video import (
                VIDEO_LOADER_REGISTRY, Qwen3VLVideoBackend, VideoSourceMetadata, VideoTargetMetadata,
            )
        except ImportError:
            self.skipTest('Run this regression in the vLLM environment as well')
        options = self.request_parameters()['media_io_kwargs']['video']
        source = VideoSourceMetadata(32, 3., 32/3)
        target = VideoTargetMetadata(options['num_frames'], options['fps'], 300)
        # The previous automatic Qwen sampler silently reduced this to 4 frames.
        self.assertEqual(Qwen3VLVideoBackend.compute_frames_index_to_sample(source, target), [0, 10, 21, 31])
        decoder = VIDEO_LOADER_REGISTRY.load(options['video_backend'])
        self.assertEqual(decoder.compute_frames_index_to_sample(source, target), list(range(32)))


if __name__ == '__main__':
    unittest.main()
