Bu faylda ikkala skriptning ham yo'riqnomasi turishi kerak. Tizimli (pipeline) mantiqqa ko'ra, avval "Batch Processor" (1-bosqich), keyin "Speaker Diarization" (2-bosqich) kelishi to'g'ri bo'ladi. README faylini ochib, Git belgilarini o'chiring va matnni quyidagi yagona formatga keltiring:

Markdown
# Audioni_tayyorlash

Ushbu repozitoriy call center audiolari asosida ASR modellarini (masalan, Whisper) o'qitish uchun ma'lumotlarni tayyorlash pipeline'ini o'z ichiga oladi.

## Requirements

- Python 3.9+
- FFmpeg installed and available on your `PATH`

Install the Python dependency:

```bash
python -m pip install -r requirements.txt
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

### 3-qadam: Konfliktni hal qilinganini Git'ga bildirish
Fayllarni to'g'ri saqlaganingizdan so'ng, terminalda quyidagi buyruqlarni ishga tushiring:

```bash
git add .
git commit -m "chore: resolve merge conflicts in README and gitignore"