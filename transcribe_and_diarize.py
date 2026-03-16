#!/usr/bin/env python3
"""
Production-ready batch transcription + speaker diarization pipeline for Gemini API.

Features:
- Reads metadata.csv with columns: file_path, original_file
- Writes/appends labeled_metadata.csv with columns: file_path, original_file, transcription
- Idempotent resume support by skipping already processed file_path values
- Immediate disk flush + fsync after each successful row append
- Exponential backoff for 429 / rate-limit / quota errors
- Uses genai.upload_file() and guarantees genai.delete_file() in finally block
"""

import argparse
import csv
import os
import random
import sys
import time
from typing import Set, Tuple

import google.generativeai as genai


SYSTEM_PROMPT = (
    "Sen Call Center audiolari bo'yicha mutaxassissan. Audioni eshit va har bir gapni "
    "kim gapirayotganini aniqlab, dialog ko'rinishida yoz. Format:\n"
    "Agent: [text]\n"
    "Mijoz: [text]\n"
    "Qoidalar: Raqamlarni so'z bilan yoz. Hech qanday qo'shimcha izoh yozma. "
    "Agar inson ovozi umuman bo'lmasa, qat'iy ravishda '[SILENCE]' deb javob ber."
)

OUTPUT_COLUMNS = ["file_path", "original_file", "transcription"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize ~60s WAV VAD chunks from metadata.csv using Gemini."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to input metadata.csv (columns: file_path, original_file).",
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-pro",
        help="Gemini model name (default: gemini-1.5-pro).",
    )
    parser.add_argument(
        "--output",
        default="labeled_metadata.csv",
        help="Output CSV path (default: labeled_metadata.csv).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum retries for rate-limit/quota/transient failures (default: 8).",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=1.5,
        help="Initial backoff delay in seconds (default: 1.5).",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=60.0,
        help="Maximum backoff delay in seconds (default: 60).",
    )
    return parser.parse_args()


def is_retryable_error(exc: Exception) -> bool:
    """
    Detect whether an exception should be retried (429/rate limit/quota/transient).
    Works defensively across potential SDK exception shapes.
    """
    text = str(exc).lower()
    retry_tokens = [
        "429",
        "rate limit",
        "quota",
        "resource exhausted",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "internal error",
        "deadline exceeded",
    ]
    return any(token in text for token in retry_tokens)


def load_processed_paths(output_csv: str) -> Set[str]:
    """
    Load already-processed file_path values from labeled_metadata.csv (if exists)
    to support idempotent resume.
    """
    processed = set()
    if not os.path.exists(output_csv):
        return processed

    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # If malformed or missing expected column, fail fast for safety.
        if reader.fieldnames is None or "file_path" not in reader.fieldnames:
            raise ValueError(
                f"Existing output file '{output_csv}' is missing required 'file_path' column."
            )
        for row in reader:
            fp = (row.get("file_path") or "").strip()
            if fp:
                processed.add(fp)
    return processed


def ensure_output_header(output_csv: str) -> None:
    """Create output file with header if it doesn't exist."""
    if os.path.exists(output_csv):
        return
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def append_row_fsync(output_csv: str, file_path: str, original_file: str, transcription: str) -> None:
    """
    Append one row and force it to disk immediately (flush + fsync).
    This guarantees survival on crash/power-loss as much as OS/filesystem allows.
    """
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(
            {
                "file_path": file_path,
                "original_file": original_file,
                "transcription": transcription,
            }
        )
        f.flush()
        os.fsync(f.fileno())


def transcribe_with_retry(
    model_name: str,
    audio_path: str,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> str:
    """
    Upload audio, request transcription+diarization, and always delete uploaded file.
    Retries retryable errors with exponential backoff + jitter.
    """
    attempt = 0
    while True:
        uploaded = None
        try:
            # Upload file to Gemini temporary storage.
            uploaded = genai.upload_file(path=audio_path)

            # Build model per call for robustness and clarity.
            model = genai.GenerativeModel(model_name=model_name)

            # Send exact prompt + audio file in a single request.
            response = model.generate_content([SYSTEM_PROMPT, uploaded])

            text = (response.text or "").strip()
            if not text:
                # Non-retryable content issue: keep explicit sentinel for observability.
                return "[EMPTY_RESPONSE]"
            return text

        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not is_retryable_error(exc):
                raise

            # Exponential backoff with bounded jitter.
            exp_delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0, exp_delay * 0.25)
            sleep_s = exp_delay + jitter
            print(
                f"[WARN] Retryable error for '{audio_path}' (attempt {attempt}/{max_retries}): {exc}. "
                f"Sleeping {sleep_s:.2f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

        finally:
            # Critical cleanup to avoid AI Studio storage buildup.
            if uploaded is not None:
                try:
                    # SDK supports delete by object or name, object is preferred when available.
                    genai.delete_file(uploaded.name if hasattr(uploaded, "name") else uploaded)
                except Exception as del_exc:
                    # Do not mask original failure; just log.
                    print(
                        f"[WARN] Failed to delete uploaded file for '{audio_path}': {del_exc}",
                        file=sys.stderr,
                    )


def validate_input_columns(fieldnames: Tuple[str, ...]) -> None:
    """Ensure metadata.csv has required columns."""
    required = {"file_path", "original_file"}
    missing = required.difference(set(fieldnames or []))
    if missing:
        raise ValueError(
            f"Input metadata is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(fieldnames or [])}"
        )


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in environment."
        )

    # Configure Gemini client.
    genai.configure(api_key=api_key)

    # Prepare output and load idempotent state.
    ensure_output_header(args.output)
    processed_paths = load_processed_paths(args.output)

    total = 0
    skipped = 0
    done = 0
    failed = 0

    with open(args.metadata, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        validate_input_columns(tuple(reader.fieldnames or []))

        for row in reader:
            total += 1
            file_path = (row.get("file_path") or "").strip()
            original_file = (row.get("original_file") or "").strip()

            if not file_path:
                print(f"[WARN] Row {total}: empty file_path, skipping.", file=sys.stderr)
                skipped += 1
                continue

            # Idempotent skip for already processed items.
            if file_path in processed_paths:
                skipped += 1
                continue

            if not os.path.exists(file_path):
                print(f"[ERROR] File does not exist: {file_path}", file=sys.stderr)
                failed += 1
                continue

            try:
                transcription = transcribe_with_retry(
                    model_name=args.model,
                    audio_path=file_path,
                    max_retries=args.max_retries,
                    base_delay=args.base_delay,
                    max_delay=args.max_delay,
                )
                append_row_fsync(args.output, file_path, original_file, transcription)
                processed_paths.add(file_path)
                done += 1
                print(f"[OK] Processed: {file_path}", file=sys.stderr)

            except Exception as exc:
                failed += 1
                print(f"[ERROR] Failed: {file_path} | {exc}", file=sys.stderr)

    print(
        f"[SUMMARY] total_rows={total}, processed={done}, skipped={skipped}, failed={failed}, "
        f"output='{args.output}'",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
