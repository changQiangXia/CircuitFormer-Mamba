from pathlib import Path
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.circuitnet import Circuitnet
from data.data_interface import collate_fn
from model.circuitformer import CircuitFormer

SAMPLE_NAME = "5607-RISCY-FPU-a-3-c5-u0.8-m1-p7-f1.npy"


def build_temporary_sample(root: Path):
    feature_dir = root / "feature"
    label_dir = root / "label"
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    feature = {
        "inst_a": np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32),
        "inst_b": np.array([30.0, 40.0, 70.0, 100.0], dtype=np.float32),
        "inst_c": np.array([15.0, 35.0, 45.0, 60.0], dtype=np.float32),
    }
    label = np.random.RandomState(0).rand(256, 256).astype(np.float32)

    np.save(feature_dir / SAMPLE_NAME, feature, allow_pickle=True)
    np.save(label_dir / SAMPLE_NAME, label)
    return feature_dir, label_dir


def run_dataset_smoke_test():
    tmp_root = Path(tempfile.mkdtemp(prefix="circuitformer_smoke_", dir=str(REPO_ROOT)))
    try:
        feature_dir, label_dir = build_temporary_sample(tmp_root)
        dataset = Circuitnet(split={"split": "train"}, data_root=feature_dir, label_root=label_dir)
        dataset.data_list = [SAMPLE_NAME]
        dataset.data_idx = np.arange(len(dataset.data_list))

        sample = dataset[0]
        assert len(sample) == 6
        batch = collate_fn([sample, sample])
        assert batch[0].shape[0] == 6
        assert batch[4].tolist() == [3, 6]
        assert batch[5].shape == (2, 1, 256, 256)
        assert batch[6].shape == (2, 1, 256, 256)
        print("dataset_smoke_test: OK")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_model_init_test():
    os.environ.setdefault("WANDB_MODE", "disabled")
    model = CircuitFormer()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model_init_test: OK, parameters={param_count}")


if __name__ == "__main__":
    run_dataset_smoke_test()
    run_model_init_test()
