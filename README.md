# Audioni_tayyorlash

Chunk `.wav` audio files into speaker-based training samples using diarization JSON metadata.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Chunk matching `.wav` files from one directory using pyannote `.json` files from another:

```bash
python chunk_audio.py \
  --audio-dir /path/to/audio_dir \
  --json-dir /path/to/json_dir \
  --output-chunks-dir /path/to/output_chunks_dir \
  --metadata-path /path/to/metadata.csv
```

The script writes chunks named like `original_speaker_0.wav`, force-splits speaker turns longer than 25 seconds into sequential 25-second pieces, skips final chunks shorter than 1 second, shows progress with `tqdm`, and generates `metadata.csv` with `file_path`, `speaker`, and `original_file` columns.
