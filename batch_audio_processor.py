from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

try:
    from tqdm import tqdm as tqdm_progress
except ImportError:  # pragma: no cover - exercised through CLI validation
    tqdm_progress = None


SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


@dataclass(frozen=True)
class ConversionResult:
    source_path: str
    output_path: str
    success: bool
    error: Optional[str] = None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def collect_audio_files(input_dir: Path, excluded_dir: Optional[Path] = None) -> list[Path]:
    audio_files: list[Path] = []
    excluded_dir = excluded_dir.resolve() if excluded_dir else None

    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        resolved_path = path.resolve()
        if excluded_dir and _is_relative_to(resolved_path, excluded_dir):
            continue

        audio_files.append(path)

    return sorted(audio_files)


def build_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = input_path.relative_to(input_dir)
    return output_dir / relative_path.with_suffix(".wav")


def convert_audio_file(source_path: str, output_path: str) -> ConversionResult:
    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
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
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        error_message = getattr(exc, "stderr", None) or str(exc)
        return ConversionResult(
            source_path=str(source),
            output_path=str(output),
            success=False,
            error=error_message.strip() or exc.__class__.__name__,
        )

    return ConversionResult(source_path=str(source), output_path=str(output), success=True)


def write_error_log(error_log_path: Path, failures: Sequence[ConversionResult]) -> None:
    if not failures:
        return

    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with error_log_path.open("a", encoding="utf-8") as error_log:
        for failure in failures:
            error_log.write(f"{failure.source_path}\t{failure.error or 'Unknown error'}\n")


def _resolve_progress_factory(
    progress_factory: Optional[Callable[..., Iterable[object]]],
) -> Callable[..., Iterable[object]]:
    if progress_factory is not None:
        return progress_factory
    if tqdm_progress is None:
        raise RuntimeError(
            "tqdm is required to display progress bars. Install dependencies from requirements.txt."
        )
    return tqdm_progress


def process_directory(
    input_dir: Path,
    output_dir: Path,
    error_log_path: Path,
    workers: Optional[int] = None,
    progress_factory: Optional[Callable[..., Iterable[object]]] = None,
    executor_cls: type[ProcessPoolExecutor] = ProcessPoolExecutor,
    completion_iterator: Callable[[Iterable[object]], Iterable[object]] = as_completed,
) -> tuple[int, int]:
    progress = _resolve_progress_factory(progress_factory)
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    excluded_dir = output_dir if _is_relative_to(output_dir, input_dir) else None

    files = collect_audio_files(input_dir, excluded_dir=excluded_dir)
    if not files:
        return 0, 0

    failures: list[ConversionResult] = []
    max_workers = workers or os.cpu_count() or 1

    with executor_cls(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                convert_audio_file,
                str(source_path),
                str(build_output_path(source_path, input_dir, output_dir)),
            )
            for source_path in files
        ]

        for future in progress(
            completion_iterator(futures),
            total=len(futures),
            desc="Processing audio files",
            unit="file",
        ):
            result = future.result()
            if not result.success:
                failures.append(result)

    write_error_log(error_log_path, failures)
    return len(files), len(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert audio files to 16kHz, 16-bit PCM mono WAV files."
    )
    parser.add_argument("input_dir", help="Directory containing source audio files.")
    parser.add_argument("output_dir", help="Directory where standardized WAV files will be written.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to all CPU cores.",
    )
    parser.add_argument(
        "--error-log",
        default=None,
        help="Path to the error log file. Defaults to <output_dir>/error_logs.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    error_log_path = Path(args.error_log).expanduser() if args.error_log else output_dir / "error_logs.txt"

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    if output_dir.resolve() == input_dir.resolve():
        raise SystemExit("Output directory must be different from the input directory.")

    processed_count, failed_count = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        error_log_path=error_log_path,
        workers=args.workers,
    )

    if processed_count == 0:
        print("No supported audio files were found.")
        return 0

    success_count = processed_count - failed_count
    print(
        f"Completed processing {processed_count} file(s): "
        f"{success_count} succeeded, {failed_count} failed."
    )
    if failed_count:
        print(f"Failed files were logged to: {error_log_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
