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
from model.model_interface import MInterface


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
        raise RuntimeError("CUDA is required for the current true Mamba one-step smoke test.")

    device = torch.device("cuda")
    cfg = OmegaConf.create(
        {
            "model_name": "circuitformer",
            "loss": "mse",
            "loss_weight": 128,
            "label_weight": 1,
            "batch_size": 2,
            "lr": 1e-4,
            "weight_decay": 0,
            "lr_scheduler": None,
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

    tmp_root = Path(tempfile.mkdtemp(prefix="true_mamba_one_step_", dir=str(REPO_ROOT)))
    try:
        feature_dir, label_dir = build_temporary_sample(tmp_root)
        dataset = Circuitnet(split={"split": "train"}, data_root=feature_dir, label_root=label_dir)
        dataset.data_list = [SAMPLE_NAME]
        dataset.data_idx = np.arange(len(dataset.data_list))
        batch = collate_fn([dataset[0], dataset[0]])
        batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)

        module = MInterface(cfg).to(device).train()
        module.log = lambda *args, **kwargs: None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = module.training_step(batch, 0)
        torch.cuda.synchronize(device)
        step_time = time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_time = time.perf_counter() - start

        info = {
            "neck_module": module.model.bev_neck.__class__.__name__,
            "loss": round(float(loss.detach().item()), 6),
            "step_time_sec": round(step_time, 6),
            "backward_time_sec": round(backward_time, 6),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
            "neck_param_count": sum(p.numel() for p in module.model.bev_neck.parameters()),
            "total_param_count": sum(p.numel() for p in module.model.parameters()),
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
