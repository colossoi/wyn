import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('capture_server', Path(__file__).resolve().parents[1] / 'scripts/capture-server.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def png_header(width=1024, height=1024):
    return b'\x89PNG\r\n\x1a\n' + (13).to_bytes(4, 'big') + b'IHDR' + width.to_bytes(4, 'big') + height.to_bytes(4, 'big')


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name) / 'video.mp4'
        self.capture = module.Capture(self.output)
        self.process = Mock(stdin=io.BytesIO())
        self.process.poll.return_value = None
        self.process.wait.return_value = 0
        self.popen = patch.object(module.subprocess, 'Popen', return_value=self.process)
        self.popen.start()

    def tearDown(self):
        self.capture.abort()
        self.popen.stop()
        self.directory.cleanup()

    def start(self, frames=3):
        self.capture.start({'totalFrames': frames, 'fps': 30})

    def test_existing_output_is_preserved(self):
        self.output.write_bytes(b'existing video')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.start()
        self.assertEqual(self.output.read_bytes(), b'existing video')
        module.subprocess.Popen.assert_not_called()

    def test_frame_requires_active_capture(self):
        with self.assertRaisesRegex(ValueError, 'No active'):
            self.capture.append(0, png_header())

    def test_sequence_rejects_missing_and_repeated_frames(self):
        self.start()
        with self.assertRaisesRegex(ValueError, 'expected 0'):
            self.capture.append(1, png_header())
        self.assertEqual(self.capture.append(0, png_header()), {'frame': 0})
        size = len(self.process.stdin.getvalue())
        with self.assertRaisesRegex(ValueError, 'expected 1'):
            self.capture.append(0, png_header())
        self.assertEqual(len(self.process.stdin.getvalue()), size)

    def test_invalid_images_do_not_reach_encoder(self):
        self.start()
        for data in [b'bad', png_header(512), png_header(height=512)]:
            with self.assertRaises(ValueError):
                self.capture.append(0, data)
        self.assertEqual(self.process.stdin.getvalue(), b'')
        self.assertEqual(self.capture.hashes, [])

    def test_incomplete_capture_cannot_be_finalized(self):
        self.start()
        self.capture.append(0, png_header())
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.capture.finish({})
        self.assertFalse(self.process.stdin.closed)
        self.assertFalse(self.output.exists())

    def test_abort_terminates_encoder_and_removes_partial(self):
        self.start()
        self.capture.partial.write_bytes(b'partial video')
        self.capture.abort()
        self.process.terminate.assert_called_once()
        self.assertTrue(self.process.stdin.closed)
        self.assertFalse(self.capture.partial.exists())
        self.assertFalse(self.output.exists())

    def test_bad_encoded_frame_count_is_not_published(self):
        self.start(1)
        self.capture.append(0, png_header())
        self.capture.partial.write_bytes(b'encoded video')
        probe = {'streams': [{'width': 1024, 'height': 1024, 'nb_read_frames': '0', 'avg_frame_rate': '30/1'}], 'format': {'duration': '0'}}
        with patch.object(module.subprocess, 'check_output', return_value=json.dumps(probe).encode()):
            with self.assertRaisesRegex(RuntimeError, 'verification'):
                self.capture.finish({})
        self.assertFalse(self.output.exists())

    def test_identical_source_images_are_preserved_as_separate_frames(self):
        self.start(3)
        for i in range(3):
            self.capture.append(i, png_header())
        self.capture.partial.write_bytes(b'encoded video')
        probe = {'streams': [{'width': 1024, 'height': 1024, 'nb_read_frames': '3', 'avg_frame_rate': '30/1'}], 'format': {'duration': '0.1'}}
        with patch.object(module.subprocess, 'check_output', return_value=json.dumps(probe).encode()):
            result = self.capture.finish({'drags': []})
        self.assertEqual(result['frames'], 3)
        report = json.loads(self.output.with_suffix('.json').read_text())
        self.assertEqual(report['uniqueSourceFrames'], 1)
        self.assertEqual(len(report['sourceFrameSHA256']), 3)
        self.assertEqual(self.output.read_bytes(), b'encoded video')


if __name__ == '__main__':
    unittest.main()
