# Audioni_tayyorlash

Chunk `.wav` audio files into speaker-based training samples using diarization JSON metadata.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Chunk a single audio/JSON pair:

```bash
python chunk_audio.py \
  --audio-file /path/to/audio.wav \
  --diarization-file /path/to/audio.json
```

Or process a directory that contains matching `file.wav` / `file.json` pairs:

```bash
python chunk_audio.py --input-dir /path/to/input
```

The script writes chunked audio to `dataset/audio/` by default and generates `metadata.csv` with `file_name` and `speaker_id` columns.
