#!/usr/bin/env python3

"""Transcribe WAV chunks from metadata.csv with the Gemini API."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator


DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"
STRICT_TRANSCRIPTION_PROMPT = (
    "Transcribe the audio exactly. Output ONLY the transcription text. "
    "Do not summarize. Convert numbers to words. If there is no human speech, "
    "output '[SILENCE]'."
)
METADATA_COLUMNS = ("file_path", "speaker", "original_file")
OUTPUT_COLUMNS = ("file_path", "speaker", "original_file", "transcription")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for metadata processing."""
    parser = argparse.ArgumentParser(
        description="Transcribe WAV chunks listed in metadata.csv with Gemini."
    )
    parser.add_argument(
        "metadata_csv",
        type=Path,
        help="Path to the input metadata.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to labeled_metadata.csv. Defaults next to metadata.csv.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name to use (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for retryable API failures (default: 5).",
    )
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=2.0,
        help="Initial exponential backoff delay in seconds (default: 2.0).",
    )
    return parser.parse_args()


def load_genai(api_key: str):
    """Import and configure the Gemini SDK only when needed."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise SystemExit(
            "google-generativeai is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    genai.configure(api_key=api_key)
    return genai


def iter_metadata_rows(metadata_path: Path) -> Iterator[dict[str, str]]:
    """Yield metadata rows and validate the expected columns."""
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("metadata.csv must include a header row.")

        missing_columns = [column for column in METADATA_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                "metadata.csv must contain the columns: "
                + ", ".join(METADATA_COLUMNS)
            )

        for row in reader:
            yield {column: row.get(column, "") for column in METADATA_COLUMNS}


def read_processed_file_paths(output_path: Path) -> set[str]:
    """Load already-processed file_path values to support resumable runs."""
    if not output_path.exists():
        return set()

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file_path" not in reader.fieldnames:
            return set()
        return {
            row["file_path"]
            for row in reader
            if row.get("file_path")
        }


def resolve_audio_path(metadata_path: Path, file_path: str) -> Path:
    """Resolve relative audio paths from the metadata.csv location."""
    audio_path = Path(file_path)
    if audio_path.is_absolute():
        return audio_path
    return (metadata_path.parent / audio_path).resolve()


def is_retryable_error(error: Exception) -> bool:
    """Identify common Gemini rate-limit and quota style failures."""
    error_name = error.__class__.__name__.lower()
    error_message = str(error).lower()
    retryable_markers = (
        "ratelimit",
        "too many requests",
        "resourceexhausted",
        "quota",
        "429",
    )
    return any(marker in error_name or marker in error_message for marker in retryable_markers)


def extract_transcription_text(response: object) -> str:
    """Extract plain text from a Gemini response object."""
    text = getattr(response, "text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", "")
            if isinstance(part_text, str) and part_text.strip():
                return part_text.strip()

    raise ValueError("Gemini response did not contain transcription text.")


def transcribe_audio(
    *,
    audio_path: Path,
    model: object,
    genai_module: object,
    max_retries: int,
    initial_backoff: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Upload one audio file, request a strict transcription, and clean it up."""
    backoff_seconds = initial_backoff

    for attempt in range(1, max_retries + 1):
        uploaded_file = None
        try:
            uploaded_file = genai_module.upload_file(path=str(audio_path))
            response = model.generate_content([STRICT_TRANSCRIPTION_PROMPT, uploaded_file])
            if hasattr(response, "resolve"):
                response.resolve()
            return extract_transcription_text(response)
        except Exception as exc:
            if attempt < max_retries and is_retryable_error(exc):
                print(
                    (
                        f"Retryable API error for {audio_path} "
                        f"(attempt {attempt}/{max_retries}): {exc}. "
                        f"Sleeping {backoff_seconds:.1f}s before retrying."
                    ),
                    file=sys.stderr,
                )
                sleep_fn(backoff_seconds)
                backoff_seconds *= 2
                continue
            raise
        finally:
            if uploaded_file is not None:
                try:
                    genai_module.delete_file(name=uploaded_file.name)
                except Exception as cleanup_error:
                    print(
                        f"Warning: failed to delete uploaded file for {audio_path}: {cleanup_error}",
                        file=sys.stderr,
                    )

    raise RuntimeError(f"Failed to transcribe {audio_path}")


def open_output_writer(output_path: Path) -> tuple[object, csv.DictWriter]:
    """Open labeled_metadata.csv in append mode and write the header once."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    handle = output_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
    if not file_exists or output_path.stat().st_size == 0:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    return handle, writer


def append_result(
    *,
    writer: csv.DictWriter,
    handle: object,
    row: dict[str, str],
    transcription: str,
) -> None:
    """Append one completed transcription and force it to disk immediately."""
    writer.writerow(
        {
            "file_path": row["file_path"],
            "speaker": row["speaker"],
            "original_file": row["original_file"],
            "transcription": transcription,
        }
    )
    handle.flush()
    os.fsync(handle.fileno())


def process_metadata(
    *,
    metadata_path: Path,
    output_path: Path,
    model_name: str,
    api_key: str,
    max_retries: int,
    initial_backoff: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    genai_module: object | None = None,
) -> tuple[int, int, int]:
    """Process metadata.csv sequentially with immediate durable writes."""
    genai = genai_module if genai_module is not None else load_genai(api_key)
    if genai_module is not None:
        genai.configure(api_key=api_key)

    processed_file_paths = read_processed_file_paths(output_path)
    model = genai.GenerativeModel(model_name)
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    handle, writer = open_output_writer(output_path)
    try:
        for row in iter_metadata_rows(metadata_path):
            file_path = row["file_path"]
            if file_path in processed_file_paths:
                skipped_count += 1
                continue

            audio_path = resolve_audio_path(metadata_path, file_path)
            if not audio_path.is_file():
                failed_count += 1
                print(f"Audio file not found: {audio_path}", file=sys.stderr)
                continue

            try:
                transcription = transcribe_audio(
                    audio_path=audio_path,
                    model=model,
                    genai_module=genai,
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                    sleep_fn=sleep_fn,
                )
            except Exception as exc:
                failed_count += 1
                print(f"Failed to transcribe {audio_path}: {exc}", file=sys.stderr)
                continue

            append_result(
                writer=writer,
                handle=handle,
                row=row,
                transcription=transcription,
            )
            processed_file_paths.add(file_path)
            processed_count += 1
    finally:
        handle.close()

    return processed_count, skipped_count, failed_count


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    metadata_path = args.metadata_csv.resolve()
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be greater than zero.")
    if args.initial_backoff <= 0:
        raise SystemExit("--initial-backoff must be greater than zero.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY environment variable is required.")

    output_path = (
        args.output.resolve()
        if args.output
        else metadata_path.with_name("labeled_metadata.csv")
    )

    processed_count, skipped_count, failed_count = process_metadata(
        metadata_path=metadata_path,
        output_path=output_path,
        model_name=args.model,
        api_key=api_key,
        max_retries=args.max_retries,
        initial_backoff=args.initial_backoff,
    )
    print(
        (
            f"Completed transcription run: processed={processed_count}, "
            f"skipped={skipped_count}, failed={failed_count}. "
            f"Output: {output_path}"
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
