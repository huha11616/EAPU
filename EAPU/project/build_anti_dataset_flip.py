from __future__ import annotations

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build anti-expert dataset with label inversion from scored train jsonl.",
        allow_abbrev=False,
    )
    parser.add_argument("--input_path", type=str, required=True, help="Scored train jsonl path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for anti dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--use_fp", action="store_true", help="Include false positives")
    parser.add_argument("--use_fn", action="store_true", help="Include false negatives")
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
            y = dict(x)
            y.pop("_row_id", None)
            y.pop("_source", None)
            f.write(json.dumps(y, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    if not args.use_fp and not args.use_fn:
        raise ValueError("At least one of --use_fp or --use_fn must be set.")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(input_path)
    print(f"[info] loaded {len(data)} scored samples")

    fp = [x for x in data if int(x["answer"]) == 0 and int(x["_pred"]) == 1]
    fn = [x for x in data if int(x["answer"]) == 1 and int(x["_pred"]) == 0]

    selected = []
    if args.use_fp:
        for x in fp:
            y = dict(x)
            y["_source"] = "fp"
            # 反转标签
            y["answer"] = 1
            selected.append(y)

    if args.use_fn:
        for x in fn:
            y = dict(x)
            y["_source"] = "fn"
            # 反转标签
            y["answer"] = 0
            selected.append(y)

    print(f"[info] fp={len(fp)} fn={len(fn)} selected_total={len(selected)}")

    if not selected:
        raise RuntimeError("No selected error samples found.")

    random.shuffle(selected)

    dev_size = max(1, int(len(selected) * args.dev_ratio))
    dev = selected[:dev_size]
    train = selected[dev_size:]

    save_jsonl(output_dir / "train.jsonl", train)
    save_jsonl(output_dir / "dev.jsonl", dev)
    save_jsonl(output_dir / "preview.jsonl", selected[: min(200, len(selected))])

    meta = {
        "input_path": str(input_path),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "use_fp": args.use_fp,
        "use_fn": args.use_fn,
        "num_total_input": len(data),
        "num_fp": len(fp),
        "num_fn": len(fn),
        "num_selected_total": len(selected),
        "num_train": len(train),
        "num_dev": len(dev),
        "train_label_0": sum(int(x["answer"]) == 0 for x in train),
        "train_label_1": sum(int(x["answer"]) == 1 for x in train),
        "dev_label_0": sum(int(x["answer"]) == 0 for x in dev),
        "dev_label_1": sum(int(x["answer"]) == 1 for x in dev),
    }

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[done] train :", output_dir / "train.jsonl")
    print("[done] dev   :", output_dir / "dev.jsonl")
    print("[done] preview:", output_dir / "preview.jsonl")
    print("[done] meta  :", output_dir / "meta.json")


if __name__ == "__main__":
    main()