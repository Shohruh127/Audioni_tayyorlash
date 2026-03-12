import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import batch_audio_processor as processor


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class FakeExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return FakeFuture(fn(*args, **kwargs))


class BatchAudioProcessorTests(unittest.TestCase):
    def test_collect_audio_files_filters_supported_extensions_and_excludes_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = input_dir / "converted"
            (input_dir / "nested").mkdir(parents=True)
            output_dir.mkdir(parents=True)

            keep_one = input_dir / "song.mp3"
            keep_two = input_dir / "nested" / "voice.M4A"
            ignored_text = input_dir / "notes.txt"
            ignored_generated = output_dir / "song.wav"

            keep_one.write_bytes(b"data")
            keep_two.write_bytes(b"data")
            ignored_text.write_text("ignore", encoding="utf-8")
            ignored_generated.write_bytes(b"data")

            files = processor.collect_audio_files(input_dir, excluded_dir=output_dir)

            self.assertCountEqual(files, [keep_one, keep_two])

    @patch("batch_audio_processor.subprocess.run")
    def test_convert_audio_file_uses_expected_ffmpeg_command(self, run_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "track.mp3"
            output = root / "out" / "track.wav"
            source.write_bytes(b"audio")

            result = processor.convert_audio_file(str(source), str(output))

            self.assertTrue(result.success)
            run_mock.assert_called_once_with(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-vn",
                    "-sn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-acodec",
                    "pcm_s16le",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_process_directory_logs_failures_and_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            error_log = output_dir / "error_logs.txt"
            input_dir.mkdir()

            valid = input_dir / "good.mp3"
            broken = input_dir / "broken.wav"
            valid.write_bytes(b"data")
            broken.write_bytes(b"data")

            def fake_convert(source_path, output_path):
                if source_path.endswith("broken.wav"):
                    return processor.ConversionResult(
                        source_path=source_path,
                        output_path=output_path,
                        success=False,
                        error="corrupted file",
                    )
                return processor.ConversionResult(
                    source_path=source_path,
                    output_path=output_path,
                    success=True,
                )

            class ExecutorWithFakeConvert(FakeExecutor):
                def submit(self, fn, *args, **kwargs):
                    return FakeFuture(fake_convert(*args, **kwargs))

            processed, failed = processor.process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                error_log_path=error_log,
                workers=2,
                progress_factory=lambda iterable, **_: iterable,
                executor_cls=ExecutorWithFakeConvert,
                completion_iterator=lambda futures: futures,
            )

            self.assertEqual(processed, 2)
            self.assertEqual(failed, 1)
            self.assertIn("broken.wav", error_log.read_text(encoding="utf-8"))

    @patch("batch_audio_processor.subprocess.run")
    def test_convert_audio_file_returns_failure_result_for_called_process_error(self, run_mock):
        run_mock.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="invalid data found when processing input",
        )

        result = processor.convert_audio_file("/tmp/source.mp3", "/tmp/output.wav")

        self.assertFalse(result.success)
        self.assertIn("invalid data", result.error)


if __name__ == "__main__":
    unittest.main()
