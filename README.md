# Audioni_tayyorlash

## Speaker diarization script

This repository includes a GPU-aware speaker diarization script at
`scripts/speaker_diarization.py`.

### What it does

- Loads the `pyannote/speaker-diarization-3.1` pipeline from Hugging Face
- Reads a folder of standardized 16kHz `.wav` files
- Uses CUDA automatically when available, otherwise falls back to CPU
- Clears GPU and CPU memory after every file to reduce long-loop memory growth
- Skips files that already have a non-empty JSON output so interrupted runs can resume
- Logs per-file failures to `diarization_errors.txt` and continues processing
- Shows batch progress with `tqdm`
- Writes one JSON file per audio file with entries like:

```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 4.5
  }
]
```

### Authentication

Set a Hugging Face token in an environment variable before running the script.
The script requires `HF_TOKEN`.

### Example

```bash
export HF_TOKEN=your_huggingface_token
python scripts/speaker_diarization.py \
  /path/to/wav_folder \
  --output-dir /path/to/output_json
```
