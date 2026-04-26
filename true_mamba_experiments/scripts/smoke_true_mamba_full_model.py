from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the current full-model true Mamba smoke test.")

    device = torch.device("cuda")
    cfg = OmegaConf.create(
        {
            "bev_mamba": {"enabled": False},
            "true_mamba": {
                "enabled": True,
                "num_blocks": 1,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "downsample": 1,
                "bidirectional": True,
                "out_proj_init_zero": True,
            },
        }
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="true_mamba_full_model_", dir=str(REPO_ROOT)))
    try:
        feature_dir, label_dir = build_temporary_sample(tmp_root)
        dataset = Circuitnet(split={"split": "train"}, data_root=feature_dir, label_root=label_dir)
        dataset.data_list = [SAMPLE_NAME]
        dataset.data_idx = np.arange(len(dataset.data_list))
        batch = collate_fn([dataset[0], dataset[0]])

        x1, y1, x2, y2, offset, label, weight = batch
        batch_inputs = [
            x1.to(device),
            y1.to(device),
            x2.to(device),
            y2.to(device),
            offset.to(device),
        ]

        model = CircuitFormer(cfg).to(device).eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(batch_inputs)
        torch.cuda.synchronize(device)
        forward_time = time.perf_counter() - start

        info = {
            "neck_module": model.bev_neck.__class__.__name__,
            "input_shape": [list(t.shape) for t in batch_inputs],
            "output_shape": list(output.shape),
            "label_shape": list(label.shape),
            "weight_shape": list(weight.shape),
            "total_param_count": sum(p.numel() for p in model.parameters()),
            "neck_param_count": sum(p.numel() for p in model.bev_neck.parameters()),
            "forward_time_sec": round(forward_time, 6),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
