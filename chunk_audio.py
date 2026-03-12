from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydub import AudioSegment
from pydub.silence import detect_silence


MIN_CHUNK_MS = 1_000
TARGET_CHUNK_MS = 25_000
MAX_CHUNK_MS = 30_000
SPLIT_SEARCH_WINDOW_MS = 2_000
MIN_SILENCE_MS = 300


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int
    speaker_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk WAV audio using diarization JSON metadata."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing matching .wav and .json files by basename.",
    )
    parser.add_argument("--audio-file", type=Path, help="Path to a single .wav file.")
    parser.add_argument(
        "--diarization-file", type=Path, help="Path to the diarization .json file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/audio"),
        help="Directory where chunked audio files will be written.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("metadata.csv"),
        help="CSV file to generate with file_name and speaker_id columns.",
    )
    args = parser.parse_args()

    if bool(args.input_dir) == bool(args.audio_file):
        parser.error("Provide either --input-dir or --audio-file/--diarization-file.")
    if args.audio_file and not args.diarization_file:
        parser.error("--diarization-file is required when --audio-file is provided.")
    return args


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
    for key in aliases[prefix]:
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


def resolve_silence_threshold(audio: AudioSegment) -> float:
    if not math.isfinite(audio.dBFS):
        return -45.0
    return max(-50.0, min(-10.0, audio.dBFS - 16.0))


def choose_split_point(turn_audio: AudioSegment, local_start_ms: int, local_end_ms: int) -> int:
    target_ms = local_start_ms + TARGET_CHUNK_MS
    search_start = max(local_start_ms, target_ms - SPLIT_SEARCH_WINDOW_MS)
    search_end = min(local_end_ms, target_ms + SPLIT_SEARCH_WINDOW_MS)
    if search_end - search_start >= MIN_SILENCE_MS:
        window_audio = turn_audio[search_start:search_end]
        silence_ranges = detect_silence(
            window_audio,
            min_silence_len=MIN_SILENCE_MS,
            silence_thresh=resolve_silence_threshold(window_audio),
        )
        if silence_ranges:
            preferred_points = [
                search_start + ((start + end) // 2)
                for start, end in silence_ranges
                if MIN_CHUNK_MS <= search_start + ((start + end) // 2) - local_start_ms <= MAX_CHUNK_MS
            ]
            if preferred_points:
                return min(
                    preferred_points,
                    key=lambda point: (point > target_ms, abs(point - target_ms)),
                )
    return target_ms


def split_segment(segment_audio: AudioSegment) -> list[AudioSegment]:
    if len(segment_audio) < MIN_CHUNK_MS:
        return []

    chunks: list[AudioSegment] = []
    local_start_ms = 0
    local_end_ms = len(segment_audio)

    while local_end_ms - local_start_ms > TARGET_CHUNK_MS:
        split_ms = choose_split_point(segment_audio, local_start_ms, local_end_ms)
        if split_ms - local_start_ms < MIN_CHUNK_MS:
            split_ms = min(local_end_ms, local_start_ms + TARGET_CHUNK_MS)
        chunks.append(segment_audio[local_start_ms:split_ms])
        local_start_ms = split_ms

    remainder = segment_audio[local_start_ms:local_end_ms]
    if len(remainder) >= MIN_CHUNK_MS:
        chunks.append(remainder)
    return [chunk for chunk in chunks if MIN_CHUNK_MS <= len(chunk) <= MAX_CHUNK_MS]


def process_pair(
    audio_path: Path,
    diarization_path: Path,
    output_dir: Path,
    metadata_base_dir: Path,
) -> list[dict[str, str]]:
    with audio_path.open("rb") as handle:
        audio = AudioSegment.from_file(handle, format="wav")
    segments = load_segments(diarization_path)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            file_name = f"{stem}_{safe_speaker}_chunk{chunk_index}.wav"
            chunk_path = output_dir / file_name
            exported_file = chunk.export(chunk_path, format="wav")
            if exported_file is not None:
                exported_file.close()
            rows.append(
                {
                    "file_name": os.path.relpath(chunk_path, metadata_base_dir).replace(
                        os.sep, "/"
                    ),
                    "speaker_id": speaker_label,
                }
            )
    return rows


def discover_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for audio_path in sorted(input_dir.glob("*.wav")):
        diarization_path = audio_path.with_suffix(".json")
        if not diarization_path.exists():
            raise FileNotFoundError(
                f"Expected diarization file {diarization_path} for {audio_path.name}"
            )
        pairs.append((audio_path, diarization_path))
    if not pairs:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")
    return pairs


def write_metadata(metadata_path: Path, rows: Iterable[dict[str, str]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "speaker_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    metadata_base_dir = args.metadata_path.resolve().parent

    if args.input_dir:
        pairs = discover_pairs(args.input_dir.resolve())
    else:
        pairs = [(args.audio_file.resolve(), args.diarization_file.resolve())]

    all_rows: list[dict[str, str]] = []
    output_dir = args.output_dir.resolve()
    for audio_path, diarization_path in pairs:
        all_rows.extend(
            process_pair(audio_path, diarization_path, output_dir, metadata_base_dir)
        )

    write_metadata(args.metadata_path.resolve(), all_rows)
    print(f"Created {len(all_rows)} chunks and wrote metadata to {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
