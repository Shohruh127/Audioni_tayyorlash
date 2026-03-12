# Audioni_tayyorlash

Gemini-based batch transcription script for `.wav` chunks listed in `metadata.csv`.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
```

## Generate pseudo-labels

The script expects a `metadata.csv` file with the columns:

- `file_path`
- `speaker`
- `original_file`

Relative `file_path` values are resolved relative to the CSV file location. The script
appends results to `labeled_metadata.csv`, skips already-transcribed `file_path` values,
and deletes uploaded files from Gemini immediately after each request.

```bash
python generate_pseudo_labels.py /path/to/metadata.csv --output /path/to/labeled_metadata.csv
```

Useful options:

```bash
python generate_pseudo_labels.py /path/to/metadata.csv --model gemini-3.1-pro-preview
```
