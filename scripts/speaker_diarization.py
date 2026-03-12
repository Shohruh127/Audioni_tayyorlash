#!/usr/bin/env python3

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ID = "pyannote/speaker-diarization-3.1"


@dataclass
class PipelineContext:
    pipeline: Any
    device: Any
    torch_module: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization for all .wav files in a folder."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing standardized .wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where diarization JSON files will be written. Defaults to the input directory.",
    )
    parser.add_argument(
        "--cache-clear-every",
        type=int,
        default=5,
        help="Clear the CUDA cache after this many files when running on GPU.",
    )
    return parser.parse_args()


def get_huggingface_token() -> str:
    token = os.getenv("HF_TOKEN")
    if token is None:
        token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    raise EnvironmentError(
        "A Hugging Face token is required. Set HF_TOKEN or HUGGINGFACE_TOKEN."
    )


def diarization_to_records(diarization: Any) -> list[dict[str, float | str]]:
    """Convert a pyannote diarization result into JSON-serializable records."""
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


def load_pipeline(token: str) -> PipelineContext:
    import torch
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(
        MODEL_ID,
        token=token,
    )
    pipeline.to(device)
    return PipelineContext(pipeline=pipeline, device=device, torch_module=torch)


def should_clear_cache(device: Any, cache_clear_every: int, processed: int) -> bool:
    return (
        device.type == "cuda"
        and processed > 0
        and cache_clear_every > 0
        and processed % cache_clear_every == 0
    )


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


def process_folder(
    input_dir: Path,
    output_dir: Path,
    cache_clear_every: int,
) -> int:
    token = get_huggingface_token()
    pipeline_context = load_pipeline(token)

    wav_files = get_wav_files(input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for audio_path in wav_files:
        try:
            diarization = pipeline_context.pipeline(audio_path)
            records = diarization_to_records(diarization)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to diarize audio file: {audio_path}. "
                "Check that the file is a readable WAV recording supported by the local "
                f"audio backend and model pipeline. Original error: {exc}"
            ) from exc

        output_path = get_output_path(audio_path, output_dir)
        output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

        processed += 1
        if should_clear_cache(pipeline_context.device, cache_clear_every, processed):
            pipeline_context.torch_module.cuda.empty_cache()
            gc.collect()

    if pipeline_context.device.type == "cuda":
        pipeline_context.torch_module.cuda.empty_cache()
    gc.collect()
    return processed


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or args.input_dir).resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    processed = process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        cache_clear_every=args.cache_clear_every,
    )
    print(f"Processed {processed} audio file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
