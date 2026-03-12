#!/usr/bin/env python3

import argparse
import csv
import gc
import math
import string
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for audio chunks listed in metadata.csv."
    )
    parser.add_argument(
        "metadata_csv",
        type=Path,
        help="Path to the input metadata.csv file containing a file_name column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output labeled_metadata.csv file. Defaults next to metadata.csv.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to faster-whisper (default: cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of audio files to process before reloading the model (default: 16).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size used during transcription (default: 5).",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    punctuation_table = str.maketrans("", "", string.punctuation)
    no_ascii_punctuation = text.translate(punctuation_table)
    no_unicode_punctuation = "".join(
        character
        for character in no_ascii_punctuation
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(no_unicode_punctuation.lower().split())


def iter_batches(rows: Sequence[dict[str, str]], batch_size: int) -> Iterator[Sequence[dict[str, str]]]:
    for batch_start in range(0, len(rows), batch_size):
        yield rows[batch_start : batch_start + batch_size]


def resolve_audio_path(metadata_path: Path, file_name: str) -> Path:
    audio_path = Path(file_name)
    if audio_path.is_absolute():
        return audio_path
    return (metadata_path.parent / audio_path).resolve()


def get_field_value(obj: object, name: str, default: float = 0.0) -> float:
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_confidence(segments: Iterable[object], info: object) -> float:
    weighted_log_probability = 0.0
    total_weight = 0.0

    for segment in segments:
        avg_logprob = get_field_value(segment, "avg_logprob", default=float("nan"))
        if math.isnan(avg_logprob):
            continue

        start = get_field_value(segment, "start", default=0.0)
        end = get_field_value(segment, "end", default=start)
        weight = max(end - start, 0.0) or 1.0
        weighted_log_probability += avg_logprob * weight
        total_weight += weight

    if total_weight:
        return max(0.0, min(1.0, math.exp(weighted_log_probability / total_weight)))

    language_probability = get_field_value(info, "language_probability", default=0.0)
    return max(0.0, min(1.0, language_probability))


def load_model(device: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return WhisperModel("large-v3", device=device, compute_type="float16")


def release_gpu_memory() -> None:
    gc.collect()

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_batch(
    batch_rows: Sequence[dict[str, str]],
    metadata_path: Path,
    device: str,
    beam_size: int,
) -> list[dict[str, str | float]]:
    model = load_model(device)
    results: list[dict[str, str | float]] = []

    try:
        for row in batch_rows:
            file_name = row["file_name"]
            audio_path = resolve_audio_path(metadata_path, file_name)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            segments, info = model.transcribe(str(audio_path), beam_size=beam_size)
            segment_list = list(segments)
            transcription = normalize_text(" ".join(getattr(segment, "text", "") for segment in segment_list))
            confidence = calculate_confidence(segment_list, info)
            results.append(
                {
                    "file_name": file_name,
                    "transcription": transcription,
                    "confidence_score": f"{confidence:.6f}",
                }
            )
    finally:
        del model
        release_gpu_memory()

    return results


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file_name" not in reader.fieldnames:
            raise ValueError("metadata.csv must contain a file_name column.")
        return list(reader)


def write_labeled_metadata(output_path: Path, rows: Sequence[dict[str, str | float]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "transcription", "confidence_score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")

    metadata_path = args.metadata_csv.resolve()
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    output_path = args.output.resolve() if args.output else metadata_path.with_name("labeled_metadata.csv")
    rows = read_metadata(metadata_path)
    labeled_rows: list[dict[str, str | float]] = []

    for batch_number, batch_rows in enumerate(iter_batches(rows, args.batch_size), start=1):
        print(
            f"Processing batch {batch_number} containing {len(batch_rows)} file(s)...",
            file=sys.stderr,
        )
        labeled_rows.extend(
            transcribe_batch(
                batch_rows=batch_rows,
                metadata_path=metadata_path,
                device=args.device,
                beam_size=args.beam_size,
            )
        )

    write_labeled_metadata(output_path, labeled_rows)
    print(f"Wrote {len(labeled_rows)} labeled rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
