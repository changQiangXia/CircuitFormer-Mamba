from pathlib import Path
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

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


def tensor_info(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp_root = Path(tempfile.mkdtemp(prefix="true_mamba_shape_audit_", dir=str(REPO_ROOT)))
    try:
        feature_dir, label_dir = build_temporary_sample(tmp_root)
        dataset = Circuitnet(split={"split": "train"}, data_root=feature_dir, label_root=label_dir)
        dataset.data_list = [SAMPLE_NAME]
        dataset.data_idx = np.arange(len(dataset.data_list))

        sample = dataset[0]
        batch = collate_fn([sample, sample])
        model = CircuitFormer().to(device).eval()

        x1, y1, x2, y2, offset, label, weight = batch
        batch_inputs = [
            x1.to(device),
            y1.to(device),
            x2.to(device),
            y2.to(device),
            offset.to(device),
        ]

        with torch.no_grad():
            encoder_out = model.encoder(batch_inputs)
            neck_out = model.bev_neck(encoder_out)
            decoder_out = model.decoder(neck_out)

        info = {
            "device": str(device),
            "batch": {
                "x1": tensor_info(batch_inputs[0]),
                "y1": tensor_info(batch_inputs[1]),
                "x2": tensor_info(batch_inputs[2]),
                "y2": tensor_info(batch_inputs[3]),
                "offset": tensor_info(batch_inputs[4]),
                "label": tensor_info(label),
                "weight": tensor_info(weight),
            },
            "model": {
                "encoder_output_dim": model.encoder.PFN.get_output_feature_dim(),
                "encoder_out": tensor_info(encoder_out),
                "neck_module": model.bev_neck.__class__.__name__,
                "neck_out": tensor_info(neck_out),
                "decoder_out": tensor_info(decoder_out),
                "total_param_count": sum(p.numel() for p in model.parameters()),
                "neck_param_count": sum(p.numel() for p in model.bev_neck.parameters()),
            },
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
