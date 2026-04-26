from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data.data_interface import DInterface  # noqa: E402
from model.model_interface import MInterface  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose NaN-related issues from a true-Mamba checkpoint.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint path to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Validation dataloader workers.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Optional positive limit on validation batches. Use 0 for full validation set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for offline diagnosis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults next to the checkpoint.",
    )
    return parser.parse_args()


def build_cfg(batch_size: int, num_workers: int):
    data_root = (REPO_ROOT / "../datasets/CircuitNet-N28/graph_features/instance_placement_micron").resolve()
    label_root = (REPO_ROOT / "../datasets/CircuitNet-N28/training_set/congestion/label").resolve()
    return OmegaConf.create(
        {
            "data": {
                "dataset": "circuitnet",
                "data_root": str(data_root),
                "label_root": str(label_root),
                "num_workers": int(num_workers),
                "batch_size": int(batch_size),
            },
            "model": {
                "model_name": "circuitformer",
                "loss": "mse",
                "max_epochs": 100,
                "warmup_epochs": 10,
                "lr_scheduler": "cosine",
                "lr": 0.001,
                "warmup_lr": 0.000001,
                "min_lr": 0.000001,
                "weight_decay": 0,
                "batch_size": int(batch_size),
                "loss_weight": 128,
                "label_weight": 1,
                "bev_mamba": {
                    "enabled": False,
                    "num_blocks": 1,
                    "inner_dim": 64,
                    "scan_downsample": 4,
                    "dw_kernel_size": 3,
                    "out_proj_init_zero": False,
                },
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
            },
        }
    )


def tensor_finite_report(state_items):
    total_tensors = 0
    bad_tensors = []
    total_values = 0
    bad_values = 0
    for name, value in state_items:
        if not torch.is_tensor(value):
            continue
        total_tensors += 1
        total_values += int(value.numel())
        finite_mask = torch.isfinite(value)
        bad_count = int((~finite_mask).sum().item())
        if bad_count:
            bad_values += bad_count
            bad_tensors.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "nonfinite_count": bad_count,
                }
            )
    return {
        "total_tensors": total_tensors,
        "nonfinite_tensors": len(bad_tensors),
        "total_values": total_values,
        "nonfinite_values": bad_values,
        "examples": bad_tensors[:20],
    }


def optimizer_finite_report(optimizer_states):
    state_items = []
    for opt_idx, optimizer_state in enumerate(optimizer_states):
        state = optimizer_state.get("state", {})
        for param_id, slot in state.items():
            if not isinstance(slot, dict):
                continue
            for slot_name, value in slot.items():
                if torch.is_tensor(value):
                    state_items.append((f"optimizer{opt_idx}.param{param_id}.{slot_name}", value))
    return tensor_finite_report(state_items)


def safe_corrcoef(pred: np.ndarray, label: np.ndarray):
    pred_std = float(pred.std())
    label_std = float(label.std())
    if not np.isfinite(pred).all():
        return None, "pred_nonfinite", pred_std, label_std
    if not np.isfinite(label).all():
        return None, "label_nonfinite", pred_std, label_std
    if pred_std == 0.0:
        return None, "pred_zero_var", pred_std, label_std
    if label_std == 0.0:
        return None, "label_zero_var", pred_std, label_std
    corr = float(np.corrcoef(pred, label)[0, 1])
    if not np.isfinite(corr):
        return None, "corr_nonfinite", pred_std, label_std
    return corr, "ok", pred_std, label_std


def sample_name_for_index(dataset, global_index: int):
    if global_index >= len(dataset.data_idx):
        return None
    data_index = int(dataset.data_idx[global_index])
    return dataset.data_list[data_index]


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    cfg = build_cfg(batch_size=args.batch_size, num_workers=args.num_workers)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    module = MInterface(cfg.model)
    load_result = module.load_state_dict(ckpt["state_dict"], strict=False)

    model_report = tensor_finite_report(module.state_dict().items())
    optimizer_report = optimizer_finite_report(ckpt.get("optimizer_states", []))

    device = torch.device(args.device)
    module = module.to(device).eval()

    data_module = DInterface(cfg.data)
    data_module.setup("fit")
    val_loader = data_module.val_dataloader()
    val_dataset = data_module.valset

    sample_reports = []
    status_counter = Counter()
    global_index = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.limit_batches > 0 and batch_idx >= args.limit_batches:
                break
            x1, y1, x2, y2, offset, label, weight = batch
            x1 = x1.to(device)
            y1 = y1.to(device)
            x2 = x2.to(device)
            y2 = y2.to(device)
            offset = offset.to(device)
            label = label.to(device)

            output = module([x1, y1, x2, y2, offset]) / cfg.model.label_weight
            output = output.detach().float().cpu()
            label_cpu = label.detach().float().cpu()

            for inner_idx in range(output.shape[0]):
                sample_name = sample_name_for_index(val_dataset, global_index)
                global_index += 1

                pred = output[inner_idx].squeeze().clamp(0, 1).reshape(-1).numpy() * 255.0
                target = label_cpu[inner_idx].squeeze().clamp(0, 1).reshape(-1).numpy() * 255.0

                corr, status, pred_std, label_std = safe_corrcoef(pred, target)
                status_counter[status] += 1

                if status != "ok":
                    sample_reports.append(
                        {
                            "sample": sample_name,
                            "status": status,
                            "pred_std": pred_std,
                            "label_std": label_std,
                            "pred_min": float(np.nanmin(pred)) if pred.size else None,
                            "pred_max": float(np.nanmax(pred)) if pred.size else None,
                            "pred_mean": float(np.nanmean(pred)) if pred.size else None,
                            "label_min": float(np.nanmin(target)) if target.size else None,
                            "label_max": float(np.nanmax(target)) if target.size else None,
                            "label_mean": float(np.nanmean(target)) if target.size else None,
                        }
                    )

    summary = {
        "checkpoint": str(args.ckpt),
        "epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", -1)),
        "load_result": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "model_state_finite_report": model_report,
        "optimizer_state_finite_report": optimizer_report,
        "validation_scan": {
            "scanned_samples": int(sum(status_counter.values())),
            "status_counts": dict(status_counter),
            "problem_examples": sample_reports[:40],
        },
    }

    output_path = args.output
    if output_path is None:
        output_path = args.ckpt.with_name(args.ckpt.stem + "_nan_diagnosis.json")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_to={output_path}")


if __name__ == "__main__":
    main()
