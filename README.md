Markdown
# Audioni_tayyorlash

Ushbu repozitoriy call center audiolari asosida ASR modellarini (masalan, Whisper) o'qitish uchun ma'lumotlarni tayyorlash pipeline'ini o'z ichiga oladi.

## Requirements

- Python 3.9+
- FFmpeg installed and available on your `PATH`

Install the Python dependencies:

```bash
pip install -r requirements.txt
Stage 1: Batch Audio Processor
Batch audio standardization script that converts mixed audio inputs into mono 16 kHz, 16-bit PCM WAV files.

Usage
Bash
python batch_audio_processor.py /path/to/input /path/to/output
Optional arguments:

--workers N to control the number of worker processes

--error-log /path/to/error_logs.txt to override the default error log location

The script scans the input directory recursively, converts each file to .wav (16 kHz, mono, 16-bit PCM), skips already converted files (resumable), safely limits CPU usage, and logs failures.

Stage 2: Speaker Diarization
A GPU-aware speaker diarization script located at scripts/speaker_diarization.py.

What it does
Loads the pyannote/speaker-diarization-3.1 pipeline from Hugging Face.

Reads a folder of standardized 16kHz .wav files.

Uses CUDA automatically when available, otherwise falls back to CPU.

Clears GPU and CPU memory after every file to reduce long-loop memory growth.

Skips files that already have a non-empty JSON output.

Logs per-file failures to diarization_errors.txt.

Writes one JSON file per audio file with millisecond-level timestamps for speakers.

Authentication
Set a Hugging Face token in an environment variable before running the script:

Bash
export HF_TOKEN=your_huggingface_token
python scripts/speaker_diarization.py /path/to/wav_folder --output-dir /path/to/output_json
Stage 3: Audio Chunking
Chunk .wav audio files into speaker-based training samples using diarization JSON metadata.

Usage
Chunk matching .wav files from one directory using pyannote .json files from another:

Bash
python chunk_audio.py \
  --audio-dir /path/to/audio_dir \
  --json-dir /path/to/json_dir \
  --output-chunks-dir /path/to/output_chunks_dir \
  --metadata-path /path/to/metadata.csv
The script writes chunks named like original_speaker_0.wav, force-splits speaker turns longer than 25 seconds into sequential 25-second pieces, skips final chunks shorter than 1 second, shows progress with tqdm, and generates metadata.csv with file_path, speaker, and original_file columns.


---

### Keyingi qadam

Fayllarni saqlab bo'lgach, ularni yana Git'ga qo'shib, konfliktni yoping:
```bash
git add .
git commit -m "chore: resolve merge conflicts and integrate Stage 3 pipeline"