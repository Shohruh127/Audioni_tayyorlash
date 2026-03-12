from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from tqdm import tqdm


# Keep Whisper chunks comfortably under the 30-second context window.
MIN_CHUNK_MS = 1_000
TARGET_CHUNK_MS = 25_000


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int
    speaker_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice WAV audio into diarization-based chunks."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing input .wav files.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Directory containing pyannote diarization .json files.",
    )
    parser.add_argument(
        "--output-chunks-dir",
        type=Path,
        default=Path("dataset/audio"),
        help="Directory where chunked audio files will be written.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("metadata.csv"),
        help="CSV file to generate with file_path, speaker, original_file columns.",
    )
    return parser.parse_args()


def sanitize_for_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "speaker"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_timestamp(value: Any) -> int:
    if isinstance(value, (int, float)):
        return max(0, int(round(float(value) * 1000)))
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"\d+(\.\d+)?", stripped):
            return max(0, int(round(float(stripped) * 1000)))
        parts = stripped.split(":")
        if len(parts) in {2, 3}:
            seconds = 0.0
            multiplier = 1.0
            for part in reversed(parts):
                seconds += float(part) * multiplier
                multiplier *= 60.0
            return max(0, int(round(seconds * 1000)))
    raise ValueError(f"Unsupported timestamp value: {value!r}")


def extract_time(data: dict[str, Any], prefix: str) -> int | None:
    candidates = (
        prefix,
        f"{prefix}_time",
        f"{prefix}Time",
        f"{prefix}_seconds",
        f"{prefix}Seconds",
    )
    for key in candidates:
        if key in data:
            return parse_timestamp(data[key])

    ms_key_candidates = (f"{prefix}_ms", f"{prefix}Ms", f"{prefix.upper()}_MS")
    for key in ms_key_candidates:
        if key in data:
            return max(0, int(round(float(data[key]))))

    aliases = {
        "start": ("begin", "offset"),
        "end": ("finish", "stop"),
    }
    for key in aliases.get(prefix, ()):
        if key in data:
            return parse_timestamp(data[key])
    return None


def extract_speaker_id(data: dict[str, Any]) -> str | None:
    for key in ("speaker_id", "speakerId", "speaker", "label", "name", "id"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _collect_segments(payload: Any) -> list[Segment]:
    """Recursively extract diarization segments from flexible JSON structures."""
    found: list[Segment] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            start_ms = extract_time(node, "start")
            end_ms = extract_time(node, "end")
            speaker_id = extract_speaker_id(node)
            if start_ms is not None and end_ms is not None and speaker_id is not None:
                if end_ms > start_ms:
                    found.append(
                        Segment(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            speaker_id=speaker_id,
                        )
                    )
                return
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return sorted(found, key=lambda segment: (segment.start_ms, segment.end_ms))


def load_segments(diarization_path: Path) -> list[Segment]:
    payload = load_json(diarization_path)
    segments = _collect_segments(payload)
    if not segments:
        raise ValueError(f"No diarization segments found in {diarization_path}")
    return segments


def split_segment(segment_audio: AudioSegment) -> list[AudioSegment]:
    """Split a speaker turn into strict sequential 25-second chunks."""
    if len(segment_audio) < MIN_CHUNK_MS:
        return []

    chunks: list[AudioSegment] = []
    local_start_ms = 0
    total_duration_ms = len(segment_audio)

    while local_start_ms + TARGET_CHUNK_MS <= total_duration_ms:
        chunks.append(segment_audio[local_start_ms : local_start_ms + TARGET_CHUNK_MS])
        local_start_ms += TARGET_CHUNK_MS

    remainder = segment_audio[local_start_ms:total_duration_ms]
    if len(remainder) >= MIN_CHUNK_MS:
        chunks.append(remainder)
    return chunks


def process_pair(
    audio_path: Path,
    diarization_path: Path,
    output_chunks_dir: Path,
    metadata_base_dir: Path,
) -> list[dict[str, str]]:
    """Slice one WAV/JSON pair and return metadata rows for emitted chunks."""
    with audio_path.open("rb") as handle:
        audio = AudioSegment.from_file(handle, format="wav")
    segments = load_segments(diarization_path)
    output_chunks_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    chunk_indices: dict[str, int] = {}
    stem = sanitize_for_filename(audio_path.stem)

    for segment in segments:
        segment_audio = audio[segment.start_ms : min(segment.end_ms, len(audio))]
        if len(segment_audio) < MIN_CHUNK_MS:
            continue
        speaker_label = segment.speaker_id
        safe_speaker = sanitize_for_filename(speaker_label)
        for chunk in split_segment(segment_audio):
            chunk_index = chunk_indices.get(safe_speaker, 0)
            chunk_indices[safe_speaker] = chunk_index + 1
            # Use a stable, model-friendly file name for downstream training jobs.
            file_name = f"{stem}_{safe_speaker}_{chunk_index}.wav"
            chunk_path = output_chunks_dir / file_name
            exported_file = chunk.export(chunk_path, format="wav")
            if exported_file is not None:
                exported_file.close()
            rows.append(
                {
                    "file_path": os.path.relpath(chunk_path, metadata_base_dir).replace(
                        os.sep, "/"
                    ),
                    "speaker": speaker_label,
                    "original_file": audio_path.name,
                }
            )
    return rows


def discover_pairs(audio_dir: Path, json_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for audio_path in sorted(audio_dir.glob("*.wav")):
        diarization_path = json_dir / f"{audio_path.stem}.json"
        if not diarization_path.exists():
            raise FileNotFoundError(
                f"Expected diarization file {diarization_path} for {audio_path.name}"
            )
        pairs.append((audio_path, diarization_path))
    if not pairs:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    return pairs


def write_metadata(metadata_path: Path, rows: list[dict[str, str]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "speaker", "original_file"])
        writer.writeheader()
        writer.writerows(rows)


def append_metadata_row(metadata_path: Path, row: dict[str, str], write_header: bool) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "speaker", "original_file"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    metadata_base_dir = args.metadata_path.resolve().parent
    pairs = discover_pairs(args.audio_dir.resolve(), args.json_dir.resolve())

    # Recreate the metadata file on each run so repeated executions stay deterministic.
    metadata_path = args.metadata_path.resolve()
    if metadata_path.exists():
        metadata_path.unlink()

    created_count = 0
    write_header = True
    output_chunks_dir = args.output_chunks_dir.resolve()
    for audio_path, diarization_path in tqdm(pairs, desc="Chunking audio files"):
        for row in process_pair(
            audio_path, diarization_path, output_chunks_dir, metadata_base_dir
        ):
            append_metadata_row(metadata_path, row, write_header)
            write_header = False
            created_count += 1

    if write_header:
        write_metadata(metadata_path, [])
    print(f"Created {created_count} chunks and wrote metadata to {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
