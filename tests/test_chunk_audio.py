import csv
import json
import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from chunk_audio import process_pair, write_metadata


class ChunkAudioTests(unittest.TestCase):
    def test_process_pair_splits_long_turns_and_ignores_short_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "session.wav"
            diarization_path = temp_path / "session.json"
            output_dir = temp_path / "dataset" / "audio"

            long_turn = (
                Sine(220).to_audio_segment(duration=24_000)
                + AudioSegment.silent(duration=1_000)
                + Sine(220).to_audio_segment(duration=10_000)
            )
            short_turn = Sine(330).to_audio_segment(duration=900)
            audio = long_turn + short_turn
            exported_file = audio.export(audio_path, format="wav")
            if exported_file is not None:
                exported_file.close()

            diarization = [
                {"start": 0.0, "end": 35.0, "speaker": "speaker_A"},
                {"start": 35.0, "end": 35.9, "speaker": "speaker_B"},
            ]
            diarization_path.write_text(json.dumps(diarization), encoding="utf-8")

            rows = process_pair(audio_path, diarization_path, output_dir, temp_path)
            self.assertEqual(2, len(rows))

            chunk_paths = [temp_path / row["file_name"] for row in rows]
            self.assertTrue(all(path.exists() for path in chunk_paths))
            lengths = []
            for path in chunk_paths:
                with path.open("rb") as handle:
                    lengths.append(len(AudioSegment.from_file(handle, format="wav")))

            self.assertTrue(24_500 <= lengths[0] <= 25_500)
            self.assertTrue(9_500 <= lengths[1] <= 11_500)
            self.assertEqual(["speaker_A", "speaker_A"], [row["speaker_id"] for row in rows])

    def test_write_metadata_uses_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.csv"
            rows = [{"file_name": "dataset/audio/a.wav", "speaker_id": "spk1"}]

            write_metadata(metadata_path, rows)

            with metadata_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                loaded_rows = list(reader)

            self.assertEqual(["file_name", "speaker_id"], reader.fieldnames)
            self.assertEqual(rows, loaded_rows)


if __name__ == "__main__":
    unittest.main()
