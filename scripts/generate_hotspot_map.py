from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.circuitnet import Circuitnet
from data.data_interface import collate_fn
from model.circuitformer import CircuitFormer


def build_cfg(variant: str):
    cfg = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    cfg.data.data_root = str(REPO_ROOT.parent / "datasets" / "CircuitNet-N28" / "graph_features" / "instance_placement_micron")
    cfg.data.label_root = str(REPO_ROOT.parent / "datasets" / "CircuitNet-N28" / "training_set" / "congestion" / "label")
    cfg.data.num_workers = 0
    cfg.data.batch_size = 1
    cfg.trainer.devices = [0]

    cfg.model.bev_mamba.enabled = False
    cfg.model.true_mamba.enabled = False

    if variant == "zero_init_mamba":
        cfg.model.bev_mamba.enabled = True
        cfg.model.bev_mamba.num_blocks = 1
        cfg.model.bev_mamba.inner_dim = 64
        cfg.model.bev_mamba.scan_downsample = 4
        cfg.model.bev_mamba.dw_kernel_size = 3
        cfg.model.bev_mamba.out_proj_init_zero = True
    elif variant == "true_mamba_scheme_b":
        cfg.model.true_mamba.enabled = True
        cfg.model.true_mamba.num_blocks = 1
        cfg.model.true_mamba.d_state = 16
        cfg.model.true_mamba.d_conv = 4
        cfg.model.true_mamba.expand = 2
        cfg.model.true_mamba.downsample = 4
        cfg.model.true_mamba.bidirectional = True
        cfg.model.true_mamba.use_input_norm = False
        cfg.model.true_mamba.use_mask = True
        cfg.model.true_mamba.mask_pool_mode = "max"
        cfg.model.true_mamba.out_proj_init_zero = False
        cfg.model.true_mamba.out_proj_init_std = 0.001
        cfg.model.true_mamba.use_residual_scale = True
        cfg.model.true_mamba.residual_scale_init = 0.001
        cfg.model.true_mamba.remask_after_upsample = True
    elif variant != "baseline":
        raise ValueError(f"Unsupported variant: {variant}")
    return cfg


def load_sample(sample_name: str):
    cfg = build_cfg("baseline")
    dataset = Circuitnet(
        split={"split": "test"},
        data_root=cfg.data.data_root,
        label_root=cfg.data.label_root,
    )
    try:
        idx = dataset.data_list.index(sample_name)
    except ValueError as exc:
        raise ValueError(f"Sample not found in test split: {sample_name}") from exc
    batch = collate_fn([dataset[idx]])
    x1, y1, x2, y2, offset, target, weight = batch
    return (x1, y1, x2, y2, offset), target.squeeze().numpy()


def load_model_prediction(ckpt_path: Path, variant: str, model_inputs, device: torch.device):
    cfg = build_cfg(variant)
    model = CircuitFormer(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    state_dict = {key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        x1, y1, x2, y2, offset = model_inputs
        pred = model([
            x1.to(device),
            y1.to(device),
            x2.to(device),
            y2.to(device),
            offset.to(device),
        ])
    return pred.squeeze().detach().cpu().numpy()


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-12 or b_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def topk_mask(arr: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    flat = arr.reshape(-1)
    k = max(1, int(np.ceil(flat.size * ratio)))
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[idx] = True
    return mask.reshape(arr.shape)


def compute_metrics(label: np.ndarray, pred: np.ndarray) -> dict:
    abs_err = np.abs(pred - label)
    gt_hot = topk_mask(label, ratio=0.05)
    pred_hot = topk_mask(pred, ratio=0.05)
    overlap = float(np.logical_and(gt_hot, pred_hot).sum() / gt_hot.sum())
    return {
        "pearson": safe_pearson(label, pred),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - label) ** 2))),
        "err_p95": float(np.quantile(abs_err, 0.95)),
        "hot_overlap": overlap,
    }


def compute_focus_window(
    label: np.ndarray,
    quantile: float = 0.995,
    min_size: int = 96,
    pad: int = 18,
):
    mask = label >= np.quantile(label, quantile)
    coords = np.argwhere(mask)
    if coords.size == 0:
        max_y, max_x = np.unravel_index(np.argmax(label), label.shape)
        coords = np.array([[max_y, max_x]])

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    h = max(int(y1 - y0) + 2 * pad, min_size)
    w = max(int(x1 - x0) + 2 * pad, min_size)
    cy = int((y0 + y1) // 2)
    cx = int((x0 + x1) // 2)

    half_h = h // 2
    half_w = w // 2

    top = max(0, cy - half_h)
    left = max(0, cx - half_w)
    bottom = min(label.shape[0], top + h)
    right = min(label.shape[1], left + w)

    top = max(0, bottom - h)
    left = max(0, right - w)
    return int(top), int(bottom), int(left), int(right)


def save_comparison_figure(sample_name: str, label, predictions: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data_blocks = [("Ground Truth", label)] + list(predictions.items())
    vmin = min(float(block.min()) for _, block in data_blocks)
    vmax = max(float(block.max()) for _, block in data_blocks)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.2), constrained_layout=True)
    flat_axes = axes.flatten()
    for ax, (title, arr) in zip(flat_axes, data_blocks):
        im = ax.imshow(arr, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=flat_axes, shrink=0.82, pad=0.02)
    cbar.set_label("Congestion intensity", fontsize=10)

    stats_lines = []
    for title, arr in data_blocks:
        stats_lines.append(
            f"{title}: max={arr.max():.3f}, mean={arr.mean():.3f}, p99={np.quantile(arr, 0.99):.3f}"
        )
    fig.suptitle(
        "Congestion Hotspot Map Comparison\n"
        f"sample={sample_name}\n" + " | ".join(stats_lines),
        fontsize=13,
    )

    out_path = out_dir / f"{Path(sample_name).stem}_hotspot_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_diagnostic_figure(sample_name: str, label, predictions: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {title: compute_metrics(label, pred) for title, pred in predictions.items()}
    errors = {title: np.abs(pred - label) for title, pred in predictions.items()}
    focus_top, focus_bottom, focus_left, focus_right = compute_focus_window(label)
    hotspot_mask = topk_mask(label, ratio=0.05).astype(np.float32)

    map_vmin = 0.0
    map_vmax = max(float(label.max()), *(float(pred.max()) for pred in predictions.values()))
    err_vmax = max(float(np.quantile(err, 0.995)) for err in errors.values())

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(17.5, 13.6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )

    top_titles = [("Ground Truth", label)] + list(predictions.items())
    map_im = None
    for col, (title, arr) in enumerate(top_titles):
        ax = axes[0, col]
        map_im = ax.imshow(arr, cmap="turbo", vmin=map_vmin, vmax=map_vmax)
        rect = Rectangle(
            (focus_left, focus_top),
            focus_right - focus_left,
            focus_bottom - focus_top,
            linewidth=1.5,
            edgecolor="white",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        if title == "Ground Truth":
            ax.set_title(
                f"{title}\nmax={arr.max():.3f}, mean={arr.mean():.3f}, p99={np.quantile(arr, 0.99):.3f}",
                fontsize=11,
            )
        else:
            stat = metrics[title]
            ax.set_title(
                f"{title}\nPearson={stat['pearson']:.3f}, MAE={stat['mae']:.3f}, overlap@5%={stat['hot_overlap']:.3f}",
                fontsize=11,
            )
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1, 0].imshow(hotspot_mask, cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Ground Truth hotspot mask\nTop-5% intensity pixels", fontsize=11)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    err_im = None
    for col, (title, err) in enumerate(errors.items(), start=1):
        ax = axes[1, col]
        err_im = ax.imshow(err, cmap="magma", vmin=0.0, vmax=err_vmax)
        stat = metrics[title]
        ax.set_title(
            f"{title} absolute error\nmean={stat['mae']:.3f}, rmse={stat['rmse']:.3f}, p95={stat['err_p95']:.3f}",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for col, (title, arr) in enumerate(top_titles):
        ax = axes[2, col]
        zoom = arr[focus_top:focus_bottom, focus_left:focus_right]
        ax.imshow(zoom, cmap="turbo", vmin=map_vmin, vmax=map_vmax)
        ax.set_title(f"{title} hotspot zoom", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    row_labels = [
        "Full-map prediction",
        "Hotspot mask and error",
        "Zoom near dominant hotspot",
    ]
    for row, label_text in enumerate(row_labels):
        axes[row, 0].set_ylabel(label_text, fontsize=11)

    fig.colorbar(map_im, ax=axes[[0, 2], :], shrink=0.84, pad=0.02, label="Congestion intensity")
    fig.colorbar(err_im, ax=axes[1, 1:], shrink=0.84, pad=0.02, label="Absolute error")
    fig.suptitle(
        "Congestion Hotspot Diagnostics\n"
        f"sample={sample_name} | zoom window: y={focus_top}:{focus_bottom}, x={focus_left}:{focus_right}\n"
        "Titles report single-sample metrics. overlap@5% denotes shared top-5% hotspot pixels.",
        fontsize=14,
    )

    out_path = out_dir / f"{Path(sample_name).stem}_hotspot_diagnostics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate a hotspot map comparison from archived checkpoints.")
    parser.add_argument("--sample-name", required=True, help="Test split sample filename, e.g. 5429-....npy")
    parser.add_argument(
        "--out-dir",
        default="analysis/hotspot_maps_2026-05-16",
        help="Output directory for the generated figure.",
    )
    args = parser.parse_args()

    model_inputs, label = load_sample(args.sample_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_map = {
        "Strong baseline": (
            REPO_ROOT / "exp" / "congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC" / "epoch=99-pearson=0.6488.ckpt",
            "baseline",
        ),
        "zero-init BEV Mamba": (
            REPO_ROOT / "exp" / "congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC" / "epoch=99-pearson=0.6499.ckpt",
            "zero_init_mamba",
        ),
        "true Mamba Scheme B": (
            REPO_ROOT / "exp" / "congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC" / "epoch=99-pearson=0.6467.ckpt",
            "true_mamba_scheme_b",
        ),
    }

    predictions = {}
    for title, (ckpt_path, variant) in ckpt_map.items():
        predictions[title] = load_model_prediction(ckpt_path, variant, model_inputs, device)

    comparison_path = save_comparison_figure(args.sample_name, label, predictions, REPO_ROOT / args.out_dir)
    diagnostic_path = save_diagnostic_figure(args.sample_name, label, predictions, REPO_ROOT / args.out_dir)
    print(comparison_path)
    print(diagnostic_path)


if __name__ == "__main__":
    main()
