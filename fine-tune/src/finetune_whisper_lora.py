"""
Fine-tuning of Whisper Tiny for gender fairness.
4 conditions: baseline, balanced, weighted, prompt_tuning

Optimized for CPU runs:
- whisper-tiny (4x faster than small)
- LoRA rank r=4 (2x fewer params than r=8)
- Only last 2 encoder layers fine-tuned
- Prompt tuning as a 4th ultra-fast condition

Usage:
    python finetune_whisper_lora.py --condition baseline
    python finetune_whisper_lora.py --condition balanced
    python finetune_whisper_lora.py --condition weighted
    python finetune_whisper_lora.py --condition prompt_tuning
"""

import os
import fire
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import get_peft_model, LoraConfig
import evaluate

TARGET_SAMPLING_RATE = 16_000
MODEL_NAME = "openai/whisper-tiny"
LANG = "en_us"
TASK = "transcribe"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch


def get_balanced_indices(dataset):
    """Return indices for gender-balanced sampling (equal male/female)."""
    df = dataset.to_pandas()
    female_idx = df[df["gender"] == 1].index.tolist()
    male_idx = df[df["gender"] == 0].index.tolist()
    min_count = min(len(female_idx), len(male_idx))
    print(f"Balancing: {min_count} female + {min_count} male = {min_count*2} total")
    return female_idx[:min_count] + male_idx[:min_count]


def compute_metrics(pred, processor):
    wer_metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def build_lora_model(model):
    """
    LoRA: inject small trainable rank-4 matrices into the last 2
    encoder attention layers only. Targets q_proj and v_proj which
    control what the model attends to in the audio signal.
    Total trainable params: ~600K out of 39M (1.5%)
    """
    lora_config = LoraConfig(
        r=4,                                  # rank 4 (was 8) = 2x fewer params
        lora_alpha=8,                         # keep alpha/r ratio = 2
        target_modules=["q_proj", "v_proj"],  # attention query + value only
        layers_to_transform=[2, 3],           # last 2 of 4 encoder layers
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_prompt_tuning_model(model):
    """
    Prompt tuning: learns 20 virtual tokens prepended to encoder input.
    ALL model weights are frozen. Only ~7680 parameters trained total.
    Much faster than LoRA but less expressive.
    """
    for param in model.parameters():
        param.requires_grad = False

    prompt_length = 20
    embed_dim = model.config.d_model  # 384 for whisper-tiny
    model.prompt_embeddings = torch.nn.Embedding(prompt_length, embed_dim)
    torch.nn.init.normal_(model.prompt_embeddings.weight, std=0.02)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}")
    return model


def main(
    condition: str = "baseline",
    num_train_epochs: int = 1,
    batch_size: int = 1,
    output_base_dir: str = "results/finetuned",
    num_workers: int = 0,
    max_train_samples: int = 200,
):
    assert condition in ("baseline", "balanced", "weighted", "prompt_tuning"), \
        "condition must be: baseline, balanced, weighted, or prompt_tuning"

    output_dir = f"{output_base_dir}/{condition}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Running condition: {condition.upper()} ===")
    print(f"Model: {MODEL_NAME} | Samples: {max_train_samples} | Epochs: {num_train_epochs}\n")

    # 1. Load processor and model
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="english", task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 2. Apply PEFT method
    if condition in ("baseline", "balanced", "weighted"):
        model = build_lora_model(model)
    elif condition == "prompt_tuning":
        model = build_prompt_tuning_model(model)

    # 3. Load FLEURS dataset
    print("Loading FLEURS...")
    train_data = load_dataset("google/fleurs", LANG, split="train", trust_remote_code=True)
    eval_data = load_dataset("google/fleurs", LANG, split="test", trust_remote_code=True)

    train_data = train_data.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))
    eval_data = eval_data.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))

    # 4. Limit training samples for CPU
    train_data = train_data.select(range(min(max_train_samples, len(train_data))))
    print(f"Training on {len(train_data)} samples")

    # 5. Gender-balanced sampling for balanced condition
    if condition == "balanced":
        balanced_idx = get_balanced_indices(train_data)
        train_data = train_data.select(balanced_idx)
        print(f"After balancing: {len(train_data)} samples")

    # 6. Preprocess audio -> features
    print("Preprocessing audio...")
    train_data = train_data.map(
        lambda b: prepare_dataset(b, processor),
        num_proc=1,
        desc="Preparing train set",
    )
    eval_data = eval_data.map(
        lambda b: prepare_dataset(b, processor),
        num_proc=1,
        desc="Preparing eval set",
    )

    # 7. Weighted loss trainer (penalizes female errors 2x)
    class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            genders = inputs.pop("gender", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if genders is not None:
                weights = torch.where(
                    genders == 1,
                    torch.tensor(2.0),  # female = 2x weight
                    torch.tensor(1.0),  # male = normal weight
                ).to(loss.device)
                loss = (loss * weights.mean()).mean()
            return (loss, outputs) if return_outputs else loss

    # 8. Training config
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=20,
        num_train_epochs=num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=num_workers,
        fp16=False,
        report_to="none",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    TrainerClass = WeightedSeq2SeqTrainer if condition == "weighted" else Seq2SeqTrainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor.feature_extractor,
    )

    # 9. Train
    print(f"Starting training for condition: {condition}...")
    trainer.train()

    # 10. Save model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nDone. Model saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)