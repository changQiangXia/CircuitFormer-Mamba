#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


def get_bin_idx(x, max_value):
    return min(int(x * np.float32(max_value)), max_value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-root",
        default=str(REPO_ROOT.parent / "datasets" / "CircuitNet-N28" / "training_set" / "congestion" / "label"),
        help="Directory that stores congestion label .npy files.",
    )
    parser.add_argument(
        "--split-file",
        default=str(REPO_ROOT / "data" / "train.txt"),
        help="Training split file used to enumerate samples.",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=1000,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "lds.txt"),
        help="Path to the output histogram file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    label_root = Path(args.label_root)
    split_file = Path(args.split_file)
    output_path = Path(args.output)

    label_list = []
    with split_file.open("r") as f:
        for line in f:
            label_list.append(line.split(",")[0].split("/")[-1].strip())

    value_dict = {x: 0 for x in range(args.max_value)}
    for label_name in tqdm(label_list):
        label = np.load(label_root / label_name).flatten()
        for value in label:
            value_dict[min(get_bin_idx(value, args.max_value), args.max_value - 1)] += 1

    output_path.write_text(json.dumps(list(value_dict.values()), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
