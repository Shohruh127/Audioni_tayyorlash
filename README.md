# Audioni_tayyorlash

Batch ASR pseudo-label generation script for audio chunks listed in `metadata.csv`.

## Setup

```bash
pip install -r requirements.txt
```

## Generate pseudo-labels

The script expects a `metadata.csv` file with a `file_name` column that points to `.wav`
chunks. Relative paths are resolved relative to the CSV file location.

```bash
python generate_pseudo_labels.py /path/to/metadata.csv --output /path/to/labeled_metadata.csv
```

By default the script:

- uses `faster-whisper` with the `large-v3` model
- runs with `compute_type="float16"`
- lowercases the transcription and removes punctuation
- writes `file_name`, `transcription`, and `confidence_score`
- reloads the model after each logical batch to reduce long-running GPU memory buildup

Useful options:

```bash
python generate_pseudo_labels.py /path/to/metadata.csv --batch-size 8 --device cuda
```
