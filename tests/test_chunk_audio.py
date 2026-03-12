import csv
import json
import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from chunk_audio import discover_pairs, process_pair, write_metadata


class ChunkAudioTests(unittest.TestCase):
    def test_process_pair_splits_strictly_at_25_seconds_and_ignores_short_chunks(self) -> None:
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
            try:
                pass
            finally:
                if exported_file is not None:
                    exported_file.close()

            diarization = [
                {"start": 0.0, "end": 35.0, "speaker": "speaker_A"},
                {"start": 35.0, "end": 35.9, "speaker": "speaker_B"},
            ]
            diarization_path.write_text(json.dumps(diarization), encoding="utf-8")

            rows = process_pair(audio_path, diarization_path, output_dir, temp_path)
            self.assertEqual(2, len(rows))

            chunk_paths = [temp_path / row["file_path"] for row in rows]
            self.assertTrue(all(path.exists() for path in chunk_paths))
            lengths = []
            for path in chunk_paths:
                with path.open("rb") as handle:
                    lengths.append(len(AudioSegment.from_file(handle, format="wav")))

            self.assertEqual([25_000, 10_000], lengths)
            self.assertEqual(
                [
                    "dataset/audio/session_speaker_A_0.wav",
                    "dataset/audio/session_speaker_A_1.wav",
                ],
                [row["file_path"] for row in rows],
            )
            self.assertEqual(["speaker_A", "speaker_A"], [row["speaker"] for row in rows])
            self.assertEqual(["session.wav", "session.wav"], [row["original_file"] for row in rows])

    def test_discover_pairs_reads_audio_and_json_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_dir = temp_path / "audio"
            json_dir = temp_path / "json"
            audio_dir.mkdir()
            json_dir.mkdir()
            (audio_dir / "sample.wav").write_bytes(b"RIFF")
            (json_dir / "sample.json").write_text("[]", encoding="utf-8")

            self.assertEqual(
                [(audio_dir / "sample.wav", json_dir / "sample.json")],
                discover_pairs(audio_dir, json_dir),
            )

    def test_write_metadata_uses_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.csv"
            rows = [
                {
                    "file_path": "dataset/audio/a.wav",
                    "speaker": "spk1",
                    "original_file": "original.wav",
                }
            ]

            write_metadata(metadata_path, rows)

            with metadata_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                loaded_rows = list(reader)

            self.assertEqual(["file_path", "speaker", "original_file"], reader.fieldnames)
            self.assertEqual(rows, loaded_rows)


if __name__ == "__main__":
    unittest.main()
