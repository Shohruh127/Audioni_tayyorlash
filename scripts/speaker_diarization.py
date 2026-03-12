#!/usr/bin/env python3

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


MODEL_ID = "pyannote/speaker-diarization-3.1"
ERROR_LOG_FILENAME = "diarization_errors.txt"


@dataclass
class PipelineContext:
    pipeline: Any
    device: Any
    torch_module: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization for all 16kHz WAV files in a folder."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing standardized 16kHz .wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where diarization JSON files will be written. Defaults to the input directory.",
    )
    return parser.parse_args()


def get_huggingface_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    raise RuntimeError("HF_TOKEN environment variable is required for diarization.")


def diarization_to_records(diarization: Any) -> list[dict[str, float | str]]:
    """Convert a pyannote Annotation into JSON-serializable speaker turns."""
    records = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        records.append(
            {
                "speaker": speaker,
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
            }
        )
    return records


def get_output_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}.json"


def output_exists_and_not_empty(output_path: Path) -> bool:
    return output_path.exists() and output_path.stat().st_size > 0


def append_error_log(error_log_path: Path, audio_path: Path, exc: Exception) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with error_log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{audio_path.name}: {exc.__class__.__name__}: {exc}\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_pipeline(token: str) -> PipelineContext:
    import torch
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline.to(device)
    return PipelineContext(pipeline=pipeline, device=device, torch_module=torch)


def clear_memory(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    gc.collect()


def get_wav_files(input_dir: Path) -> list[Path]:
    wav_files = sorted(
        audio_file
        for audio_file in input_dir.iterdir()
        if audio_file.is_file() and audio_file.suffix.lower() == ".wav"
    )
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    stems = {}
    for audio_path in wav_files:
        existing = stems.get(audio_path.stem)
        if existing is not None:
            raise ValueError(
                f"Multiple audio files map to the same output name: {existing.name} and {audio_path.name}"
            )
        stems[audio_path.stem] = audio_path

    return wav_files


def iter_with_progress(
    items: list[Path],
    progress_factory: Callable[..., Iterable[Path]] | None = None,
) -> Iterable[Path]:
    if progress_factory is not None:
        return progress_factory(items, total=len(items), desc="Diarizing", unit="file")

    from tqdm import tqdm

    return tqdm(items, total=len(items), desc="Diarizing", unit="file")


def process_folder(
    input_dir: Path,
    output_dir: Path,
    pipeline_context: PipelineContext | None = None,
    progress_factory: Callable[..., Iterable[Path]] | None = None,
    error_log_path: Path | None = None,
) -> tuple[int, int, int]:
    # Load the pipeline once and keep it resident on the best available device.
    pipeline_context = pipeline_context or load_pipeline(get_huggingface_token())
    wav_files = get_wav_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = error_log_path or output_dir / ERROR_LOG_FILENAME

    processed = 0
    skipped = 0
    failed = 0

    for audio_path in iter_with_progress(wav_files, progress_factory=progress_factory):
        output_path = get_output_path(audio_path, output_dir)
        try:
            # Skip files that already have a non-empty JSON result for resumable batch runs.
            if output_exists_and_not_empty(output_path):
                skipped += 1
                continue

            diarization = pipeline_context.pipeline(audio_path)
            records = diarization_to_records(diarization)
            output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            processed += 1
        except Exception as exc:
            # Persist the failure immediately so long runs can continue without losing context.
            append_error_log(error_log_path, audio_path, exc)
            failed += 1
        finally:
            # Force VRAM and CPU memory cleanup after every file to reduce long-loop leaks.
            clear_memory(pipeline_context.torch_module, pipeline_context.device)

    return processed, skipped, failed


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or args.input_dir).resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    processed, skipped, failed = process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
    )
    print(
        f"Completed diarization. processed={processed}, skipped={skipped}, failed={failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
