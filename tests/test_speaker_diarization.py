import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(
    __file__
).resolve().parents[1] / "scripts" / "speaker_diarization.py"


def load_module():
    spec = importlib.util.spec_from_file_location("speaker_diarization", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self, segments):
        self._segments = segments

    def itertracks(self, yield_label=False):
        for start, end, speaker in self._segments:
            yield FakeTurn(start, end), None, speaker


class FakePipeline:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def __call__(self, audio_path):
        name = Path(audio_path).name
        self.calls.append(name)
        response = self._responses[name]
        if isinstance(response, Exception):
            raise response
        return response


class FakeTorchCuda:
    def __init__(self):
        self.empty_cache_calls = 0

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorchModule:
    def __init__(self):
        self.cuda = FakeTorchCuda()


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type


class SpeakerDiarizationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_get_huggingface_token_requires_hf_token(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                self.module.get_huggingface_token()

        with mock.patch.dict("os.environ", {"HF_TOKEN": "secret"}, clear=True):
            self.assertEqual(self.module.get_huggingface_token(), "secret")

    def test_process_folder_skips_existing_outputs_logs_errors_and_clears_memory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            (input_dir / "a.wav").write_bytes(b"")
            (input_dir / "b.wav").write_bytes(b"")
            (input_dir / "c.WAV").write_bytes(b"")

            output_dir.mkdir()
            (output_dir / "a.json").write_text("[{\"speaker\":\"SPEAKER_00\"}]\n", encoding="utf-8")

            fake_pipeline = FakePipeline(
                {
                    "b.wav": FakeAnnotation([(0.0, 4.5, "SPEAKER_00")]),
                    "c.WAV": RuntimeError("CUDA out of memory"),
                }
            )
            fake_torch = FakeTorchModule()
            pipeline_context = self.module.PipelineContext(
                pipeline=fake_pipeline,
                device=FakeDevice("cuda"),
                torch_module=fake_torch,
            )

            with mock.patch.object(self.module.gc, "collect") as mock_collect:
                processed, skipped, failed = self.module.process_folder(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    pipeline_context=pipeline_context,
                    progress_factory=lambda items, **_: items,
                )

            self.assertEqual((processed, skipped, failed), (1, 1, 1))
            self.assertEqual(fake_pipeline.calls, ["b.wav", "c.WAV"])
            total_files = processed + skipped + failed
            self.assertEqual(fake_torch.cuda.empty_cache_calls, total_files)
            self.assertEqual(mock_collect.call_count, total_files)

            records = json.loads((output_dir / "b.json").read_text(encoding="utf-8"))
            self.assertEqual(
                records,
                [{"speaker": "SPEAKER_00", "start": 0.0, "end": 4.5}],
            )

            error_log = (output_dir / self.module.ERROR_LOG_FILENAME).read_text(
                encoding="utf-8"
            )
            self.assertIn("c.WAV", error_log)
            self.assertIn("CUDA out of memory", error_log)

    def test_process_folder_clears_memory_for_failed_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            (input_dir / "broken.wav").write_bytes(b"")

            fake_pipeline = FakePipeline({"broken.wav": ValueError("corrupt wav")})
            fake_torch = FakeTorchModule()
            pipeline_context = self.module.PipelineContext(
                pipeline=fake_pipeline,
                device=FakeDevice("cuda"),
                torch_module=fake_torch,
            )

            with mock.patch.object(self.module.gc, "collect") as mock_collect:
                processed, skipped, failed = self.module.process_folder(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    pipeline_context=pipeline_context,
                    progress_factory=lambda items, **_: items,
                )

            self.assertEqual((processed, skipped, failed), (0, 0, 1))
            self.assertEqual(fake_torch.cuda.empty_cache_calls, 1)
            self.assertEqual(mock_collect.call_count, 1)


if __name__ == "__main__":
    unittest.main()
