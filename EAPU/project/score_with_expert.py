from __future__ import annotations

import json
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--expert_adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_row_id"] = i
            rows.append(obj)
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def batched(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    cleanup()

    data = load_jsonl(input_path)
    print(f"[info] loaded {len(data)} samples from {input_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        use_fast=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=2,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    expert_model = PeftModel.from_pretrained(
        base_model,
        args.expert_adapter_path,
        local_files_only=True,
    )
    expert_model.eval()

    out_rows = []

    with torch.no_grad():
        for batch in batched(data, args.batch_size):
            texts = [f"Comment: {x['input']}" for x in batch]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_length,
            ).to(expert_model.device)

            logits = expert_model(**inputs).logits.float()
            probs = F.softmax(logits, dim=-1)

            for i, x in enumerate(batch):
                y = dict(x)
                y["_pred"] = int(torch.argmax(logits[i]).item())
                y["_prob_0"] = float(probs[i, 0].item())
                y["_prob_1"] = float(probs[i, 1].item())
                y["_margin_to_boundary"] = abs(y["_prob_1"] - 0.5)
                out_rows.append(y)

            del inputs, logits, probs

    save_jsonl(output_path, out_rows)

    del expert_model
    del base_model
    cleanup()

    print(f"[done] wrote scored dataset to: {output_path}")


if __name__ == "__main__":
    main()