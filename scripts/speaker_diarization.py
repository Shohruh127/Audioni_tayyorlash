#!/usr/bin/env python3

import argparse
import gc
import json
import os
from pathlib import Path


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
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    raise EnvironmentError(
        "A Hugging Face token is required. Set HF_TOKEN or HUGGINGFACE_TOKEN."
    )


def diarization_to_records(diarization) -> list[dict[str, float | str]]:
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


def load_pipeline(token: str):
    import torch
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    pipeline.to(device)
    return pipeline, device, torch


def process_folder(
    input_dir: Path,
    output_dir: Path,
    cache_clear_every: int,
) -> int:
    token = get_huggingface_token()
    pipeline, device, torch = load_pipeline(token)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for audio_path in wav_files:
        diarization = pipeline(str(audio_path))
        records = diarization_to_records(diarization)

        output_path = get_output_path(audio_path, output_dir)
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        processed += 1
        if device.type == "cuda" and cache_clear_every > 0 and processed % cache_clear_every == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
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
