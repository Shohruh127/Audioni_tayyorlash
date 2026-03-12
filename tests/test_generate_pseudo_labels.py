import csv
import tempfile
import unittest
from pathlib import Path

import generate_pseudo_labels as script


class FakeUploadedFile:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeRateLimitError(Exception):
    pass


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, parts):
        self.calls.append(parts)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


class FakeGenAI:
    def __init__(self, responses):
        self.responses = responses
        self.configure_calls = []
        self.upload_calls = []
        self.delete_calls = []
        self.models = []

    def configure(self, api_key: str):
        self.configure_calls.append(api_key)

    def upload_file(self, path: str):
        self.upload_calls.append(path)
        return FakeUploadedFile(name=f"uploaded::{Path(path).name}", path=path)

    def delete_file(self, name: str):
        self.delete_calls.append(name)

    def GenerativeModel(self, model_name: str):
        model = FakeModel(self.responses)
        model.model_name = model_name
        self.models.append(model)
        return model


class GeneratePseudoLabelsTests(unittest.TestCase):
    def test_process_metadata_skips_existing_rows_and_appends_new_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metadata_path = temp_path / "metadata.csv"
            output_path = temp_path / "labeled_metadata.csv"
            chunk_one = temp_path / "chunk1.wav"
            chunk_two = temp_path / "chunk2.wav"
            chunk_one.write_bytes(b"one")
            chunk_two.write_bytes(b"two")

            metadata_path.write_text(
                "file_path,speaker,original_file\n"
                "chunk1.wav,spk1,orig.wav\n"
                "chunk2.wav,spk2,orig.wav\n",
                encoding="utf-8",
            )
            output_path.write_text(
                "file_path,speaker,original_file,transcription\n"
                "chunk1.wav,spk1,orig.wav,done already\n",
                encoding="utf-8",
            )

            fake_genai = FakeGenAI(["fresh transcript"])
            processed_count, skipped_count, failed_count = script.process_metadata(
                metadata_path=metadata_path,
                output_path=output_path,
                model_name="gemini-test",
                api_key="test-key",
                max_retries=3,
                initial_backoff=1.0,
                genai_module=fake_genai,
            )

            self.assertEqual((processed_count, skipped_count, failed_count), (1, 1, 0))
            self.assertEqual(fake_genai.configure_calls, ["test-key"])
            self.assertEqual(fake_genai.upload_calls, [str(chunk_two.resolve())])
            self.assertEqual(fake_genai.delete_calls, ["uploaded::chunk2.wav"])
            self.assertEqual(fake_genai.models[0].model_name, "gemini-test")
            prompt, uploaded_file = fake_genai.models[0].calls[0]
            self.assertEqual(prompt, script.STRICT_TRANSCRIPTION_PROMPT)
            self.assertEqual(uploaded_file.path, str(chunk_two.resolve()))

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["transcription"], "done already")
            self.assertEqual(rows[1]["file_path"], "chunk2.wav")
            self.assertEqual(rows[1]["transcription"], "fresh transcript")

    def test_transcribe_audio_retries_rate_limit_and_deletes_each_upload(self):
        fake_genai = FakeGenAI([FakeRateLimitError("quota exceeded"), "retry success"])
        model = fake_genai.GenerativeModel("gemini-test")
        audio_path = Path("/tmp/fake.wav")
        sleep_calls = []

        transcription = script.transcribe_audio(
            audio_path=audio_path,
            model=model,
            genai_module=fake_genai,
            max_retries=2,
            initial_backoff=2.0,
            sleep_fn=sleep_calls.append,
        )

        self.assertEqual(transcription, "retry success")
        self.assertEqual(sleep_calls, [2.0])
        self.assertEqual(fake_genai.upload_calls, [str(audio_path), str(audio_path)])
        self.assertEqual(
            fake_genai.delete_calls,
            ["uploaded::fake.wav", "uploaded::fake.wav"],
        )


if __name__ == "__main__":
    unittest.main()
