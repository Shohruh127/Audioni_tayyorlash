# Audioni_tayyorlash

Batch audio standardization script that converts mixed audio inputs into mono 16 kHz, 16-bit PCM WAV files.

## Requirements

- Python 3.9+
- FFmpeg installed and available on your `PATH`

Install the Python dependency:

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python batch_audio_processor.py /path/to/input /path/to/output
```

Optional arguments:

- `--workers N` to control the number of worker processes
- `--error-log /path/to/error_logs.txt` to override the default error log location

The script:

- scans the input directory recursively for supported audio files
- converts each file to `.wav` using 16 kHz sample rate, mono channel, and 16-bit PCM encoding
- preserves the relative file name/path inside the output directory
- logs failed files to `error_logs.txt` and continues processing the remaining files
- shows batch progress with `tqdm`
