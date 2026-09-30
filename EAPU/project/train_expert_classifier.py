import sys
from pathlib import Path


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


if len(sys.argv) != 4:
    print("Usage: python train_expert_classifier.py <base_model_path> <data_dir> <output_dir>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
data_dir = sys.argv[2]
output_dir = Path(sys.argv[3])

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{data_dir}/train.jsonl",
        "validation": f"{data_dir}/val.jsonl",
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def preprocess(example):
    text = f"Comment: {example['input']}"
    enc = tokenizer(text, truncation=True, max_length=512)
    enc["labels"] = int(example["answer"])
    return enc


dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="SEQ_CLS",
    modules_to_save=["score"],
)

model = get_peft_model(model, lora_config)


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


training_args = TrainingArguments(
    output_dir=str(output_dir),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=1e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

adapter_dir = output_dir / "expert_adapter_final"
adapter_dir.mkdir(parents=True, exist_ok=True)

# 保存“回载后的最优 adapter”
trainer.model.save_pretrained(str(adapter_dir))

print("✅ Best expert adapter saved:", adapter_dir)