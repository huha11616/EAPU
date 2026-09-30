from __future__ import annotations

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train anti model from base model with LoRA.",
        allow_abbrev=False,
    )
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def load_json_dataset(data_dir: str):
    data_dir = Path(data_dir)
    train_file = data_dir / "train.jsonl"
    dev_file = data_dir / "dev.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    if not dev_file.exists():
        raise FileNotFoundError(f"Missing dev file: {dev_file}")

    ds = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(dev_file),
        }
    )
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[info] loading dataset...")
    ds = load_json_dataset(args.data_dir)
    print(f"[info] train={len(ds['train'])} dev={len(ds['validation'])}")

    print("[info] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(batch: Dict[str, List[Any]]):
        texts = [f"Comment: {x}" for x in batch["input"]]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
        )
        enc["labels"] = [int(x) for x in batch["answer"]]
        return enc

    print("[info] tokenizing...")
    ds = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="tokenize",
    )

    print("[info] loading raw base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=2,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    anti_lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=None,
    )

    print("[info] attaching anti LoRA on base model...")
    model = get_peft_model(model, anti_lora_cfg)

    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[info] trainable params: {trainable:,}")
    print(f"[info] total params:     {total:,}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir)),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[info] start training anti-from-base...")
    trainer.train()

    print("[info] evaluating best anti model...")
    metrics = trainer.evaluate()
    print("[eval]", metrics)

    adapter_dir = Path(args.output_dir) / "anti_from_base_adapter_final"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print("[info] saving best anti adapter...")
    trainer.model.save_pretrained(str(adapter_dir))

    with open(Path(args.output_dir) / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_path": args.base_model_path,
                "data_dir": args.data_dir,
                "num_train": len(ds["train"]),
                "num_dev": len(ds["validation"]),
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "best_eval": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[done] best anti-from-base adapter saved to:", adapter_dir)


if __name__ == "__main__":
    main()