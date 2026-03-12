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

    @patch("batch_audio_processor.subprocess.run")
    def test_convert_audio_file_skips_existing_non_empty_output(self, run_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "track.mp3"
            output = root / "out" / "track.wav"
            source.write_bytes(b"audio")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"existing")

            result = processor.convert_audio_file(str(source), str(output))

            self.assertTrue(result.success)
            self.assertTrue(result.skipped)
            run_mock.assert_not_called()

    def test_process_directory_logs_failures_and_continues_immediately(self):
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

            with patch("batch_audio_processor.append_error_log") as append_error_log:
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
            append_error_log.assert_called_once()
            logged_failure = append_error_log.call_args.args[1]
            self.assertIn("broken.wav", logged_failure.source_path)

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

    def test_determine_worker_count_leaves_two_cpu_cores_free(self):
        with patch("batch_audio_processor.os.cpu_count", return_value=8):
            self.assertEqual(processor.determine_worker_count(None), 6)

        with patch("batch_audio_processor.os.cpu_count", return_value=1):
            self.assertEqual(processor.determine_worker_count(None), 1)

        self.assertEqual(processor.determine_worker_count(3), 3)


if __name__ == "__main__":
    unittest.main()
