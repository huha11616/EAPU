from __future__ import annotations

import sys
import csv
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

if len(sys.argv) != 5:
    print(
        "Usage: python test_classifier.py "
        "<data_dir> "
        "<base_model_path> "
        "<adapter_path> "
        "<split>"
    )
    sys.exit(1)

data_dir = sys.argv[1]
base_model_path = sys.argv[2]
adapter_path = sys.argv[3]
split = sys.argv[4]

# =====================================
# device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =====================================
# dataset
# =====================================

dataset = load_dataset(
    "json",
    data_files={
        "test":
        f"{data_dir}/{split}.jsonl"
    }
)["test"]

# =====================================
# tokenizer
# =====================================

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    use_fast=False,
    local_files_only=True
)

if tokenizer.pad_token_id is None:

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]"}
        )

# =====================================
# base model
# =====================================

print("[info] loading base")

try:

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        torch_dtype=torch.float16,
        local_files_only=True
    )

except:

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        local_files_only=True
    )

model.to(device)

model.config.pad_token_id = tokenizer.pad_token_id

# =====================================
# load adapter
# =====================================

print("[info] loading adapter")

model = PeftModel.from_pretrained(
    model,
    adapter_path,
    local_files_only=True
)

model.eval()

# =====================================
# evaluation
# =====================================

preds = []
labels = []

for example in dataset:

    text = (
        f"Comment: "
        f"{example['input']}"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        logits = model(
            **inputs
        ).logits

        pred = torch.argmax(
            logits,
            dim=-1
        ).item()

    preds.append(pred)

    labels.append(
        int(example["answer"])
    )

# =====================================
# metrics
# =====================================

acc = accuracy_score(
    labels,
    preds
)

p, r, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="binary",
    zero_division=0
)

print("\n===== Evaluation =====")

print("Accuracy :", acc)
print("Precision:", p)
print("Recall   :", r)
print("F1       :", f1)

# ========== 新增详细分类报告和混淆矩阵 ==========
print("\n===== Classification Report =====")
print(classification_report(labels, preds, digits=4))

print("\n===== Confusion Matrix =====")
print(confusion_matrix(labels, preds))
# ================================================

# =====================================
# save
# =====================================

with open(
    "results_adapter.csv",
    "a",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        adapter_path,
        acc,
        p,
        r,
        f1
    ])

print(
    "✅ Results saved to results_adapter.csv"
)