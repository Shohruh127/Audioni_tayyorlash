# Audioni_tayyorlash

This repository includes a Hugging Face/PyTorch training script for fine-tuning Whisper on a local ASR dataset.

## Dataset layout

Prepare your dataset so `load_dataset("audiofolder", data_dir=...)` can read it, for example:

```text
dataset/
├── audio/
│   ├── sample-0001.wav
│   └── sample-0002.wav
└── metadata.csv
```

The `metadata.csv` file should contain the relative audio path and transcript text columns expected by the `audiofolder` dataset loader.
If your transcript column is not named `text`, pass `--text-column your_column_name`.

## Install

```bash
pip install -r requirements.txt
```

## Fine-tune Whisper with LoRA

```bash
python fine_tune_whisper.py \
  --data-dir /path/to/dataset \
  --model-name openai/whisper-small \
  --output-dir /path/to/output
```

The script:

- loads the dataset with `load_dataset("audiofolder", data_dir=...)`
- resamples audio to 16 kHz
- prepares Whisper input features and tokenized labels with `WhisperProcessor`
- applies LoRA to the Whisper query/value attention projections via `peft`
- trains with `Seq2SeqTrainer`, mixed precision, and gradient checkpointing
- evaluates using Word Error Rate (WER)
- keeps the best checkpoint based on validation WER

If your dataset only has one split, the script automatically creates a validation split with `--validation-size`.
