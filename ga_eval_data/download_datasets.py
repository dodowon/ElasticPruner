# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

"""
Download evaluation datasets for genetic algorithm experiments.
Datasets: arc_challenge, arc_easy, hellaswag, mmlu, boolq, winogrande

Usage:
    python download_datasets.py --save_dir /path/to/save
"""

import argparse
import os
from datasets import load_dataset


DATASETS = {
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "splits": ["train", "validation", "test"],
    },
    "arc_easy": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "splits": ["train", "validation", "test"],
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "splits": ["train", "validation"],
    },
    "boolq": {
        "path": "google/boolq",
        "name": None,
        "splits": ["train", "validation"],
    },
    "winogrande": {
        "path": "allenai/winogrande",
        "name": "winogrande_xl",
        "splits": ["train", "validation"],
    },
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "splits": ["test", "validation", "dev"],
    },
}


def download_and_save(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name, cfg in DATASETS.items():
        out_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Downloading: {dataset_name}")
        print(f"  path={cfg['path']}, name={cfg['name']}")
        print(f"  save -> {out_dir}")

        try:
            load_kwargs = {"path": cfg["path"]}
            if cfg["name"] is not None:
                load_kwargs["name"] = cfg["name"]

            for split in cfg["splits"]:
                try:
                    ds = load_dataset(**load_kwargs, split=split, trust_remote_code=True)
                    split_path = os.path.join(out_dir, split)
                    ds.save_to_disk(split_path)
                    print(f"  [{split}] {len(ds)} examples -> {split_path}")
                except Exception as e:
                    print(f"  [{split}] SKIPPED: {e}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. All datasets saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ICL evaluation datasets")
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Root directory to save datasets (e.g. /path/to/ga_eval_data)",
    )
    args = parser.parse_args()
    download_and_save(args.save_dir)


if __name__ == "__main__":
    main()
