import sys
import csv
import gc
import json
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

if len(sys.argv) != 4:
    print(
        "Usage: python "
        "test_classifier_fullmodel.py "
        "<data_dir> "
        "<model_path> "
        "<split>"
    )
    sys.exit(1)

data_dir, model_path, split = sys.argv[1:4]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

dataset = load_dataset(
    "json",
    data_files={
        "test":
        f"{data_dir}/{split}.jsonl"
    }
)["test"]

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        model_path,
        use_fast=False,
        local_files_only=True
    )
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = (
        tokenizer.eos_token
    )

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        model_path,
        num_labels=2,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map="auto",
    )
)

model.config.pad_token_id = (
    tokenizer.pad_token_id
)

model.eval()

preds = []
labels = []

for example in dataset:

    # 和训练保持完全一致
    text = (
        f"Comment: "
        f"{example['input']}"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        logits = (
            model(**inputs)
            .logits
        )

        pred = torch.argmax(
            logits,
            dim=-1
        ).item()

    preds.append(pred)

    labels.append(
        int(example["answer"])
    )

acc = accuracy_score(
    labels,
    preds
)

p, r, f1, _ = (
    precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )
)

tn, fp, fn, tp = (
    confusion_matrix(
        labels,
        preds
    )
    .ravel()
)

print("\n===== Evaluation =====")

print("Accuracy:", acc)
print("Precision:", p)
print("Recall:", r)
print("F1:", f1)

print(
    "TN/FP/FN/TP:",
    tn,
    fp,
    fn,
    tp
)

# 写 csv
with open(
    "results_fullmodel.csv",
    "a",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        model_path,
        acc,
        p,
        r,
        f1,
        tn,
        fp,
        fn,
        tp
    ])

# 写 json（给总控读）
metric = {
    "accuracy": float(acc),
    "precision": float(p),
    "recall": float(r),
    "f1": float(f1),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}

with open(
    "tmp_metric.json",
    "w"
) as f:

    json.dump(
        metric,
        f
    )

print("DONE")