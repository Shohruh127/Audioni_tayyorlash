#!/usr/bin/env python3
"""Fine-tune Whisper on a local audiofolder dataset with LoRA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


DEFAULT_TARGET_MODULES = ("q_proj", "v_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Whisper model for ASR using a local dataset laid out with "
            "metadata.csv and an audio/ directory."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Path to the audiofolder dataset directory.")
    parser.add_argument(
        "--model-name",
        default="openai/whisper-small",
        choices=("openai/whisper-small", "openai/whisper-base"),
        help="Whisper checkpoint to fine-tune.",
    )
    parser.add_argument("--output-dir", default="whisper-asr-lora", help="Directory to save checkpoints.")
    parser.add_argument("--language", default=None, help="Optional language token for multilingual Whisper.")
    parser.add_argument("--task", default="transcribe", help="Tokenizer task, usually 'transcribe'.")
    parser.add_argument("--train-split", default="train", help="Dataset split to use for training.")
    parser.add_argument("--eval-split", default=None, help="Optional existing dataset split to use for validation.")
    parser.add_argument("--text-column", default="text", help="Transcript column name in metadata.csv.")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Validation ratio if eval split is absent.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Number of fine-tuning epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Trainer logging frequency.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation/checkpoint frequency.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="How many checkpoints to keep.")
    parser.add_argument("--generation-max-length", type=int, default=225, help="Max generated token length.")
    parser.add_argument("--max-label-length", type=int, default=225, help="Max label length retained during preprocessing.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


@dataclass
class WhisperDataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> Dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        starts_with_decoder_token = labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().item()
        if starts_with_decoder_token:
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class MetricComputer:
    def __init__(self, processor: Any, metric: Any) -> None:
        self.processor = processor
        self.metric = metric

    def __call__(self, eval_pred: Any) -> Dict[str, float]:
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = replace_ignored_label_ids(eval_pred.label_ids, self.processor.tokenizer.pad_token_id)

        predictions = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": float(self.metric.compute(predictions=predictions, references=references))}


def replace_ignored_label_ids(label_ids: Any, pad_token_id: int) -> Any:
    copied = label_ids.copy()
    if isinstance(copied, list):
        return [
            [pad_token_id if token_id == -100 else token_id for token_id in sequence]
            for sequence in copied
        ]
    copied[copied == -100] = pad_token_id
    return copied


def resolve_dataset_splits(dataset: Mapping[str, Any], train_split: str, eval_split: Optional[str], validation_size: float, seed: int) -> Dict[str, Any]:
    if train_split not in dataset:
        available = ", ".join(sorted(dataset))
        raise ValueError(f"Training split '{train_split}' was not found. Available splits: {available}")

    train_dataset = dataset[train_split]

    if eval_split:
        if eval_split not in dataset:
            available = ", ".join(sorted(dataset))
            raise ValueError(f"Evaluation split '{eval_split}' was not found. Available splits: {available}")
        return {"train": train_dataset, "eval": dataset[eval_split]}

    split_dataset = train_dataset.train_test_split(test_size=validation_size, seed=seed)
    return {"train": split_dataset["train"], "eval": split_dataset["test"]}


def load_runtime_dependencies() -> Dict[str, Any]:
    try:
        import evaluate
        from datasets import Audio, load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing required dependencies. Install torch, transformers, datasets, evaluate, jiwer, peft, accelerate, and soundfile."
        ) from exc

    return {
        "evaluate": evaluate,
        "Audio": Audio,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
        "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
        "WhisperProcessor": WhisperProcessor,
        "set_seed": set_seed,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")
    if not 0 < args.validation_size < 1:
        raise SystemExit("--validation-size must be strictly between 0 and 1 (exclusive).")

    runtime = load_runtime_dependencies()
    runtime["set_seed"](args.seed)

    processor = runtime["WhisperProcessor"].from_pretrained(args.model_name, language=args.language, task=args.task)
    model = runtime["WhisperForConditionalGeneration"].from_pretrained(args.model_name)

    dataset = runtime["load_dataset"]("audiofolder", data_dir=str(data_dir))
    split_dataset = resolve_dataset_splits(dataset, args.train_split, args.eval_split, args.validation_size, args.seed)

    split_dataset["train"] = split_dataset["train"].cast_column("audio", runtime["Audio"](sampling_rate=16000))
    split_dataset["eval"] = split_dataset["eval"].cast_column("audio", runtime["Audio"](sampling_rate=16000))

    def prepare_batch(batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=16000,
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_label_length,
        ).input_ids
        return batch

    train_dataset = split_dataset["train"].map(
        prepare_batch,
        remove_columns=split_dataset["train"].column_names,
        desc="Preparing training split",
    )
    eval_dataset = split_dataset["eval"].map(
        prepare_batch,
        remove_columns=split_dataset["eval"].column_names,
        desc="Preparing validation split",
    )

    decoder_prompt_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = decoder_prompt_ids
    model.config.suppress_tokens = []
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = decoder_prompt_ids
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False

    lora_config = runtime["LoraConfig"](
        task_type=runtime["TaskType"].SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(DEFAULT_TARGET_MODULES),
    )
    model = runtime["get_peft_model"](model, lora_config)

    training_args = runtime["Seq2SeqTrainingArguments"](
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    wer_metric = runtime["evaluate"].load("wer")

    trainer = runtime["Seq2SeqTrainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperDataCollator(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        tokenizer=processor.feature_extractor,
        compute_metrics=MetricComputer(processor=processor, metric=wer_metric),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
