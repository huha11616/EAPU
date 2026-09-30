from __future__ import annotations

import argparse
import re
import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        required=True
    )

    parser.add_argument(
        "--expert_adapter",
        type=str,
        required=True
    )

    parser.add_argument(
        "--anti_adapter",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--lambda_value",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--last_n_layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16"
    )

    return parser.parse_args()


def build_topk_mask(
    tensor: torch.Tensor,
    ratio: float
):
    flat = tensor.abs().flatten()

    k = max(
        1,
        int(flat.numel() * ratio)
    )

    threshold = torch.topk(
        flat,
        k
    ).values[-1]

    return (
        tensor.abs()
        >= threshold
    )


def should_modify_layer(
    name: str,
    total_layers: int,
    last_n_layers: int
):
    if "score.weight" in name:
        return True

    m = re.search(
        r"layers\.(\d+)\.",
        name
    )

    if not m:
        return False

    layer_id = int(
        m.group(1)
    )

    return (
        layer_id
        >=
        total_layers
        - last_n_layers
    )


def main():
    args = parse_args()

    dtype = (
        torch.float16
        if args.dtype == "float16"
        else torch.float32
    )

    print("[info] loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model
    )

    print("[info] loading base...")

    base_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.base_model,
            num_labels=2,
            torch_dtype=dtype,
            device_map="cpu"
        )
    )

    total_layers = (
        base_model.config
        .num_hidden_layers
    )

    print(
        "[info] total_layers:",
        total_layers
    )

    print("[info] loading expert...")

    expert_model = (
        PeftModel.from_pretrained(
            base_model,
            args.expert_adapter
        )
        .merge_and_unload()
    )

    print("[info] loading anti...")

    base_for_anti = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.base_model,
            num_labels=2,
            torch_dtype=dtype,
            device_map="cpu"
        )
    )

    anti_model = (
        PeftModel.from_pretrained(
            base_for_anti,
            args.anti_adapter
        )
        .merge_and_unload()
    )

    expert_state = (
        expert_model.state_dict()
    )

    anti_state = (
        anti_model.state_dict()
    )

    base_state = (
        base_model.state_dict()
    )

    merged_state = {}

    modified_layers = 0
    untouched_layers = 0

    for name in expert_state:

        expert_param = expert_state[name]

        if (
            not expert_param
            .is_floating_point()
        ):
            merged_state[name] = (
                expert_param
            )
            continue

        modify = should_modify_layer(
            name,
            total_layers,
            args.last_n_layers
        )

        if not modify:
            merged_state[name] = (
                expert_param
            )
            untouched_layers += 1
            continue

        anti_param = anti_state[name]
        base_param = base_state[name]

        anti_delta = (
            anti_param
            - base_param
        )

        mask = build_topk_mask(
            anti_delta,
            args.topk_ratio
        )

        sparse_anti = (
            anti_delta * mask
        )

        merged_state[name] = (
            expert_param
            - args.lambda_value
            * sparse_anti
        )

        modified_layers += 1

        print(
            f"[modify] {name}"
        )

    print(
        f"[info] modified="
        f"{modified_layers}"
    )

    print(
        f"[info] untouched="
        f"{untouched_layers}"
    )

    final_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.base_model,
            num_labels=2,
            torch_dtype=dtype,
            device_map="cpu"
        )
    )

    final_model.load_state_dict(
        merged_state,
        strict=False
    )

    final_model.save_pretrained(
        args.output_dir
    )

    tokenizer.save_pretrained(
        args.output_dir
    )

    print(
        "[done]",
        args.output_dir
    )


if __name__ == "__main__":
    main()