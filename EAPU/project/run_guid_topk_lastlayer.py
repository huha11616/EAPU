from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys

from pathlib import Path
from collections import defaultdict


ROOT = Path(
    ROOT/EAPU
).resolve()

PROJECT_DIR = (
    ROOT / "project"
)

DATA_ROOT = (
    ROOT
    / "data"
    / "expert_lopo"
)

CKPT_ROOT = (
    ROOT / "ckpt"
)

RUNS_ROOT = (
    CKPT_ROOT / "runs"
)

OUTPUT_ROOT = (
    ROOT
    / "project"
    / "output"
)


PROJECTS = [
    "apache-ant-1.7.0",
    "apache-jmeter-2.10",
    "argouml",
    "columba-1.4-src",
    "emf-2.4.1",
    "hibernate-distribution-3.3.2.GA",
    "jEdit-4.2",
    "jfreechart-1.0.19",
    "jruby-1.4.0",
    "sql12",
]


LAMBDA_LIST = [
    "0.01",
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "0.75"
]

TOPK_LIST = [
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "0.75"
]

LAST_LAYER_LIST = [
    "2",
    "4",
    "8",
    "16"
]


def run_cmd(
    cmd: list[str],
    env_extra: dict | None = None
):
    env = os.environ.copy()

    if env_extra:
        env.update(env_extra)

    print(
        "\n[cmd]",
        " ".join(cmd)
    )

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n"
            f"{' '.join(cmd)}"
        )


def main():

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python "
            "run_guid_topk_lastlayer.py "
            "<base_model_path> "
            "<tag>"
        )

        sys.exit(1)

    base_model_path = (
        sys.argv[1]
    )

    tag = sys.argv[2]

    out_root = (
        OUTPUT_ROOT
        / f"guid_topk_lastlayer_{tag}"
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True
    )

    detail_csv = (
        out_root
        / "all_results.csv"
    )

    best_csv = (
        out_root
        / "best_results.csv"
    )

    avg_csv = (
        out_root
        / "average_results.csv"
    )

    with open(
        detail_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "project",
            "lambda",
            "topk",
            "layer",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp"
        ])

    with open(
        avg_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "lambda",
            "topk",
            "layer",
            "avg_accuracy",
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_tn",
            "avg_fp",
            "avg_fn",
            "avg_tp"
        ])

    best_rows = []

    for lam in LAMBDA_LIST:
        for topk in TOPK_LIST:
            for layer in LAST_LAYER_LIST:

                print("\n")
                print("=" * 100)

                print(
                    f"[CONFIG] "
                    f"lambda={lam} "
                    f"topk={topk} "
                    f"layer={layer}"
                )

                metrics_sum = (
                    defaultdict(float)
                )

                valid_projects = 0

                for project in PROJECTS:

                    print(
                        f"\n[PROJECT] "
                        f"{project}"
                    )

                    expert_adapter = (
                        CKPT_ROOT
                        / (
                            f"expert_"
                            f"{tag}_"
                           # f"_split_"
                            f"{project}"
                        )
                        / "expert_adapter_final"
                    )

                    anti_adapter = (
                        RUNS_ROOT
                        / (
                            f"anti_"
                            f"{tag}_"
                            f"{project}"
                        )
                        / (
                            "anti_from_base_"
                            "adapter_final"
                        )
                    )

                    data_dir = (
                        DATA_ROOT
                        / f"split_{project}"
                    )

                    merged_dir = (
                        CKPT_ROOT
                        / (
                            f"tmp_guid_"
                            f"{project}"
                            f"_lam_{lam}"
                            f"_k_{topk}"
                            f"_layer_{layer}"
                        )
                    )

                    try:

                        if merged_dir.exists():
                            shutil.rmtree(
                                merged_dir
                            )

                        # =================
                        # merge
                        # =================
                        run_cmd(
                            [
                                "python",
                                str(
                                    PROJECT_DIR
                                    / "merge_topk_last_layer.py"
                                ),

                                "--base_model",
                                base_model_path,

                                "--expert_adapter",
                                str(
                                    expert_adapter
                                ),

                                "--anti_adapter",
                                str(
                                    anti_adapter
                                ),

                                "--output_dir",
                                str(
                                    merged_dir
                                ),

                                "--lambda_value",
                                lam,

                                "--topk_ratio",
                                topk,

                                "--last_n_layers",
                                layer,

                                "--dtype",
                                "float16"
                            ],

                            env_extra={
                                "CUDA_VISIBLE_DEVICES":
                                "0",

                                "PYTORCH_CUDA_ALLOC_CONF":
                                (
                                    "expandable_segments:"
                                    "True"
                                )
                            }
                        )

                        # =================
                        # test
                        # =================
                        run_cmd(
                            [
                                "python",

                                str(
                                    PROJECT_DIR
                                    / "test_classifier_fullmodel.py"
                                ),

                                str(data_dir),

                                str(
                                    merged_dir
                                ),

                                "test"
                            ],

                            env_extra={
                                "CUDA_VISIBLE_DEVICES":
                                "0",

                                "PYTORCH_CUDA_ALLOC_CONF":
                                (
                                    "expandable_segments:"
                                    "True"
                                )
                            }
                        )

                        # =================
                        # read metric
                        # =================
                        metric_path = (
                            PROJECT_DIR
                            / "tmp_metric.json"
                        )

                        import json

                        with open(
                            metric_path,
                            "r"
                        ) as f:

                            metric = (
                                json.load(f)
                            )

                        acc = metric[
                            "accuracy"
                        ]

                        p = metric[
                            "precision"
                        ]

                        r = metric[
                            "recall"
                        ]

                        f1 = metric[
                            "f1"
                        ]

                        tn = metric["tn"]
                        fp = metric["fp"]
                        fn = metric["fn"]
                        tp = metric["tp"]

                        with open(
                            detail_csv,
                            "a",
                            newline="",
                            encoding="utf-8"
                        ) as f:

                            writer = (
                                csv.writer(f)
                            )

                            writer.writerow([
                                project,
                                lam,
                                topk,
                                layer,
                                acc,
                                p,
                                r,
                                f1,
                                tn,
                                fp,
                                fn,
                                tp
                            ])

                        metrics_sum[
                            "acc"
                        ] += acc

                        metrics_sum[
                            "p"
                        ] += p

                        metrics_sum[
                            "r"
                        ] += r

                        metrics_sum[
                            "f1"
                        ] += f1

                        metrics_sum[
                            "tn"
                        ] += tn

                        metrics_sum[
                            "fp"
                        ] += fp

                        metrics_sum[
                            "fn"
                        ] += fn

                        metrics_sum[
                            "tp"
                        ] += tp

                        valid_projects += 1

                    finally:

                        # 删除模型
                        if (
                            merged_dir
                            .exists()
                        ):
                            shutil.rmtree(
                                merged_dir
                            )

                            print(
                                "[cleanup] "
                                f"deleted "
                                f"{merged_dir}"
                            )

                # ==================
                # average
                # ==================
                if valid_projects > 0:

                    avg_row = [
                        lam,
                        topk,
                        layer,

                        metrics_sum[
                            "acc"
                        ]
                        / valid_projects,

                        metrics_sum[
                            "p"
                        ]
                        / valid_projects,

                        metrics_sum[
                            "r"
                        ]
                        / valid_projects,

                        metrics_sum[
                            "f1"
                        ]
                        / valid_projects,

                        metrics_sum[
                            "tn"
                        ]
                        / valid_projects,

                        metrics_sum[
                            "fp"
                        ]
                        / valid_projects,

                        metrics_sum[
                            "fn"
                        ]
                        / valid_projects,

                        metrics_sum[
                            "tp"
                        ]
                        / valid_projects,
                    ]

                    with open(
                        avg_csv,
                        "a",
                        newline="",
                        encoding="utf-8"
                    ) as f:

                        writer = (
                            csv.writer(f)
                        )

                        writer.writerow(
                            avg_row
                        )

                    best_rows.append({
                        "lambda":
                        lam,

                        "topk":
                        topk,

                        "layer":
                        layer,

                        "avg_f1":
                        avg_row[6]
                    })

    best_rows = sorted(
        best_rows,
        key=lambda x:
        x["avg_f1"],
        reverse=True
    )

    with open(
        best_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda",
                "topk",
                "layer",
                "avg_f1"
            ]
        )

        writer.writeheader()

        writer.writerows(
            best_rows
        )

    print(
        "\n✅ GUID GRID "
        "SEARCH FINISHED."
    )


if __name__ == "__main__":
    main()