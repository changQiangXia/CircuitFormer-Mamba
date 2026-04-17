#!/usr/bin/env python3
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "analysis" / "acceptance_2026-04-15"
FIG_DIR = OUT_DIR / "figures"
sys.path.insert(0, str(REPO_ROOT))

from model.circuitformer import CircuitFormer


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path):
    return Path(path).read_text(errors="ignore")


def line_count(path):
    with Path(path).open("r") as f:
        return sum(1 for _ in f)


def parse_metric_triplets(text):
    pattern = re.compile(
        r"pearson:\s*([-+0-9.eE]+)\s+spearman:\s*([-+0-9.eE]+)\s+kendall:\s*([-+0-9.eE]+)"
    )
    results = []
    for match in pattern.finditer(text):
        results.append(
            {
                "pearson": float(match.group(1)),
                "spearman": float(match.group(2)),
                "kendall": float(match.group(3)),
            }
        )
    return results


def parse_history_from_log(path):
    metrics = parse_metric_triplets(read_text(path))
    # The first metric in train/launcher logs comes from sanity checking.
    metrics = metrics[1:]
    history = []
    for idx, metric in enumerate(metrics):
        epoch = 9 + idx * 10
        history.append({"epoch": epoch, **metric})
    return history


def parse_single_metric_from_log(path):
    metrics = parse_metric_triplets(read_text(path))
    if not metrics:
        raise ValueError(f"No metric triplet found in {path}")
    return metrics[-1]


def parse_ckpt_metric_from_name(path):
    match = re.search(r"epoch=(\d+)-pearson=([0-9.]+)\.ckpt$", str(path))
    if not match:
        raise ValueError(f"Cannot parse checkpoint name: {path}")
    return {"epoch": int(match.group(1)), "pearson": float(match.group(2))}


def parse_run_start_utc(run_name):
    return datetime.strptime(run_name, "%Y-%m-%d_%H-%M-%S_UTC").replace(tzinfo=timezone.utc)


def file_mtime_utc(path):
    ts = os.stat(path).st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def count_params():
    cfg = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    variants = [
        ("baseline", False, False),
        ("mamba", True, False),
        ("mamba_zero_init", True, True),
    ]
    results = {}
    for name, enabled, zero_init in variants:
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg2.model.bev_mamba.enabled = enabled
        cfg2.model.bev_mamba.out_proj_init_zero = zero_init
        model = CircuitFormer(cfg2.model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[name] = {"total": total, "trainable": trainable}
    return results


def build_summary():
    formal_train = REPO_ROOT / "exp" / "congestion_formal_2026-04-11_18-09-54_UTC" / "train.log"
    formal_test = REPO_ROOT / "exp" / "congestion_formal_2026-04-11_18-09-54_UTC" / "test_epoch99.log"
    formal_best = REPO_ROOT / "exp" / "congestion_formal_2026-04-11_18-09-54_UTC" / "epoch=99-pearson=0.5570.ckpt"

    rerun_train = REPO_ROOT / "exp" / "congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC" / "launcher.log"
    rerun_test = REPO_ROOT / "exp" / "congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC" / "acceptance_test_epoch99.log"
    rerun_best = REPO_ROOT / "exp" / "congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC" / "epoch=99-pearson=0.6488.ckpt"

    zero_resume = REPO_ROOT / "exp" / "congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC" / "resume_2026-04-14_16-17-12_UTC.log"
    zero_test = REPO_ROOT / "exp" / "congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC" / "acceptance_test_epoch99.log"
    zero_ckpt_99 = REPO_ROOT / "exp" / "congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC" / "epoch=99-pearson=0.6499.ckpt"

    run1_fail = REPO_ROOT / "exp" / "congestion_bev_mamba_run1_2026-04-13_09-58-40_UTC" / "launcher.log"

    formal_history = parse_history_from_log(formal_train)
    rerun_history = parse_history_from_log(rerun_train)
    zero_final_val = parse_single_metric_from_log(zero_resume)
    zero_epoch89_observed = {
        "epoch": 89,
        "pearson": 0.6475,
        "observed_utc": datetime(2026, 4, 14, 14, 47, 15, tzinfo=timezone.utc).isoformat(),
    }
    zero_history = [
        {"epoch": zero_epoch89_observed["epoch"], "pearson": zero_epoch89_observed["pearson"]},
        {"epoch": 99, "pearson": zero_final_val["pearson"]},
    ]

    params = count_params()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "train": line_count(REPO_ROOT / "data" / "train.txt"),
            "val": line_count(REPO_ROOT / "data" / "val.txt"),
            "test": line_count(REPO_ROOT / "data" / "test.txt"),
        },
        "params": params,
        "experiments": {
            "formal": {
                "label": "Original training recipe",
                "start_utc": parse_run_start_utc("2026-04-11_18-09-54_UTC").isoformat(),
                "end_utc": file_mtime_utc(formal_best).isoformat(),
                "best_ckpt": "exp/congestion_formal_2026-04-11_18-09-54_UTC/epoch=99-pearson=0.5570.ckpt",
                "val_history": formal_history,
                "final_val": formal_history[-1],
                "test": parse_single_metric_from_log(formal_test),
            },
            "rerun": {
                "label": "Strong baseline (lr=1e-3, seed=3407)",
                "start_utc": parse_run_start_utc("2026-04-12_10-58-54_UTC").isoformat(),
                "end_utc": file_mtime_utc(rerun_best).isoformat(),
                "best_ckpt": "exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC/epoch=99-pearson=0.6488.ckpt",
                "val_history": rerun_history,
                "final_val": rerun_history[-1],
                "test": parse_single_metric_from_log(rerun_test),
            },
            "zero_init_mamba": {
                "label": "BEV Mamba + zero-init",
                "start_utc": parse_run_start_utc("2026-04-13_19-08-30_UTC").isoformat(),
                "epoch89_utc": zero_epoch89_observed["observed_utc"],
                "resume_start_utc": parse_run_start_utc("2026-04-14_16-17-12_UTC").isoformat(),
                "end_utc": file_mtime_utc(zero_ckpt_99).isoformat(),
                "best_ckpt": "exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC/epoch=99-pearson=0.6499.ckpt",
                "val_history": zero_history,
                "final_val": {"epoch": 99, **zero_final_val},
                "test": parse_single_metric_from_log(zero_test),
            },
        },
        "events": [
            {
                "time_utc": parse_run_start_utc("2026-04-11_18-09-54_UTC").isoformat(),
                "title": "Original CircuitFormer run starts",
                "detail": "Official-style training recipe used as the first full reproduction.",
            },
            {
                "time_utc": parse_run_start_utc("2026-04-12_10-58-54_UTC").isoformat(),
                "title": "Strong baseline rerun starts",
                "detail": "Learning rate raised to 1e-3 and seed fixed to 3407.",
            },
            {
                "time_utc": parse_run_start_utc("2026-04-13_09-58-40_UTC").isoformat(),
                "title": "First Mamba attempt fails",
                "detail": "DataLoader hits FileNotFoundError because the dataset path points to ../datasets/... and the required npy file is absent there.",
            },
            {
                "time_utc": parse_run_start_utc("2026-04-13_10-01-21_UTC").isoformat(),
                "title": "BEV Mamba run restarts",
                "detail": "The run is relaunched after correcting the data path issue.",
            },
            {
                "time_utc": parse_run_start_utc("2026-04-13_19-08-30_UTC").isoformat(),
                "title": "Zero-init Mamba run starts",
                "detail": "The output projection of the BEV Mamba neck is initialized to zero.",
            },
            {
                "time_utc": zero_epoch89_observed["observed_utc"],
                "title": "Zero-init run reaches epoch 89",
                "detail": "A valid checkpoint is saved before the later interruption.",
            },
            {
                "time_utc": parse_run_start_utc("2026-04-14_16-17-12_UTC").isoformat(),
                "title": "Interrupted training is resumed",
                "detail": "Training resumes from last.ckpt and keeps the same save directory.",
            },
            {
                "time_utc": file_mtime_utc(zero_ckpt_99).isoformat(),
                "title": "Final Mamba checkpoint saved",
                "detail": "The resumed run reaches epoch 99 and saves the final accepted checkpoint.",
            },
        ],
        "notes": {
            "metric_definition": "Pearson, Spearman, and Kendall are computed per sample on the predicted and target congestion maps, then averaged across the dataset.",
            "validation_cadence": "Validation runs every 10 epochs according to trainer.check_val_every_n_epoch=10.",
            "zero_init_history_note": "The early stdout of the zero-init run was not preserved as a full train.log, so its intermediate curve before epoch 89 is reconstructed only from saved checkpoints and the later resume log.",
            "run1_failure_excerpt": "FileNotFoundError: [Errno 2] No such file or directory: '../datasets/CircuitNet-N28/graph_features/instance_placement_micron/6274-RISCY-FPU-b-1-c20-u0.9-m1-p4-f1.npy'",
            "project_position_note": "This project is positioned as a high-quality reproduction, engineering repair, and exploratory improvement with one Mamba-inspired lightweight BEV residual neck.",
            "method_scope_note": "The accepted BEV Mamba variant is a lightweight residual neck inspired by selective state space ideas, inserted between encoder and decoder in the current CircuitFormer pipeline.",
            "contribution_scope_note": "The largest metric jump in this project comes from the strong baseline rerun; the BEV Mamba gain versus that baseline is an exploratory small positive gain.",
            "experiment_scope_note": "Public conclusions here cover congestion prediction under the current weighted-MSE training path, using one strong-baseline run and one zero-init Mamba run. The DRC entry in config and the configurable loss field remain code-level interfaces rather than finalized experimental tracks.",
            "prior_note": "The decoder encoder loads ResNet18 pretrained weights, so the accepted route includes an external visual pretraining prior.",
            "variance_note": "The accepted comparison uses one strong-baseline run and one zero-init Mamba run, so the reported gain is a single-run observation.",
        },
    }

    zero_val = summary["experiments"]["zero_init_mamba"]["final_val"]
    rerun_val = summary["experiments"]["rerun"]["final_val"]
    zero_test_metrics = summary["experiments"]["zero_init_mamba"]["test"]
    rerun_test_metrics = summary["experiments"]["rerun"]["test"]

    gain_vs_rerun = {}
    for split_name, newer, older in [
        ("val", zero_val, rerun_val),
        ("test", zero_test_metrics, rerun_test_metrics),
    ]:
        gain_vs_rerun[split_name] = {}
        for key in ("pearson", "spearman", "kendall"):
            abs_gain = newer[key] - older[key]
            rel_gain = 100.0 * abs_gain / older[key]
            gain_vs_rerun[split_name][key] = {
                "absolute": abs_gain,
                "relative_percent": rel_gain,
            }
    summary["gain_vs_rerun"] = gain_vs_rerun

    param_gain = params["mamba_zero_init"]["total"] - params["baseline"]["total"]
    summary["param_overhead_vs_baseline"] = {
        "absolute": param_gain,
        "relative_percent": 100.0 * param_gain / params["baseline"]["total"],
    }

    return summary


def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_validation_curves(summary):
    plt.figure(figsize=(9, 5.2))
    colors = {
        "formal": "#4C78A8",
        "rerun": "#F58518",
        "zero_init_mamba": "#54A24B",
    }
    for key in ("formal", "rerun", "zero_init_mamba"):
        history = summary["experiments"][key]["val_history"]
        xs = [item["epoch"] for item in history]
        ys = [item["pearson"] for item in history]
        style = "-" if len(xs) > 2 else "--"
        plt.plot(xs, ys, marker="o", linewidth=2.2, linestyle=style, color=colors[key], label=summary["experiments"][key]["label"])
        plt.scatter(xs[-1], ys[-1], s=80, color=colors[key])
        plt.text(xs[-1] + 0.8, ys[-1], f"{ys[-1]:.4f}", fontsize=9, color=colors[key])

    plt.xlabel("Epoch")
    plt.ylabel("Validation Pearson")
    plt.title("Validation Pearson Across Reproduced Runs")
    plt.grid(alpha=0.25, linestyle="--")
    plt.xticks([9, 19, 29, 39, 49, 59, 69, 79, 89, 99])
    plt.legend(frameon=False)
    _savefig(FIG_DIR / "validation_pearson_curves.png")


def plot_metric_bars(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    exp_keys = ["formal", "rerun", "zero_init_mamba"]
    metrics = ["pearson", "spearman", "kendall"]
    x = list(range(len(exp_keys)))
    width = 0.22
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for ax, split_name, getter in [
        (axes[0], "Validation", lambda key: summary["experiments"][key]["final_val"]),
        (axes[1], "Test", lambda key: summary["experiments"][key]["test"]),
    ]:
        for idx, metric in enumerate(metrics):
            values = [getter(key)[metric] for key in exp_keys]
            xpos = [value + (idx - 1) * width for value in x]
            ax.bar(xpos, values, width=width, label=metric.capitalize(), color=colors[idx])
            for xp, yp in zip(xpos, values):
                ax.text(xp, yp + 0.003, f"{yp:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([summary["experiments"][key]["label"] for key in exp_keys], rotation=12, ha="right")
        ax.set_title(f"{split_name} Metrics")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_ylim(0.30, 0.725)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Final Validation/Test Metrics", y=0.99)
    fig.subplots_adjust(top=0.74)
    _savefig(FIG_DIR / "final_metric_bars.png")


def plot_gain_vs_rerun(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    metrics = ["pearson", "spearman", "kendall"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for ax, split_name in zip(axes, ["val", "test"]):
        gains = [summary["gain_vs_rerun"][split_name][metric]["absolute"] for metric in metrics]
        bars = ax.bar(metrics, gains, color=colors)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Zero-init Mamba minus strong baseline ({split_name})")
        ax.set_ylabel("Absolute metric gain")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for bar, value, metric in zip(bars, gains, metrics):
            rel = summary["gain_vs_rerun"][split_name][metric]["relative_percent"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.0004,
                f"{value:+.4f}\n({rel:+.2f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ymax = max(gains) * 1.6 if max(gains) > 0 else 0.01
        ax.set_ylim(0.0, ymax)

    fig.suptitle("Where the Mamba gain shows up")
    _savefig(FIG_DIR / "gain_vs_rerun.png")


def plot_dataset_and_params(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    split_names = ["train", "val", "test"]
    split_values = [summary["dataset"][name] for name in split_names]
    axes[0].bar(split_names, split_values, color=["#4C78A8", "#72B7B2", "#E45756"])
    axes[0].set_title("Dataset split sizes")
    axes[0].set_ylabel("Samples")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    for idx, value in enumerate(split_values):
        axes[0].text(idx, value + max(split_values) * 0.015, str(value), ha="center", fontsize=9)

    param_names = ["baseline", "mamba_zero_init"]
    param_values = [summary["params"][name]["total"] / 1e6 for name in param_names]
    axes[1].bar(["Baseline", "Mamba+zero-init"], param_values, color=["#F58518", "#54A24B"])
    axes[1].set_title("Model size comparison")
    axes[1].set_ylabel("Parameters (millions)")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    for idx, value in enumerate(param_values):
        axes[1].text(idx, value + 0.05, f"{value:.3f}M", ha="center", fontsize=9)

    overhead = summary["param_overhead_vs_baseline"]
    axes[1].text(
        0.5,
        min(param_values) + 0.20,
        f"+{overhead['absolute']:,} params\n(+{overhead['relative_percent']:.3f}%)",
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F3F3F3", "edgecolor": "#AAAAAA"},
    )

    fig.suptitle("Data scale and parameter overhead")
    _savefig(FIG_DIR / "dataset_and_params.png")


def plot_timeline(summary):
    fig, ax = plt.subplots(figsize=(12.2, 4.8))

    rows = [
        ("formal", 2.5, "#4C78A8"),
        ("rerun", 1.5, "#F58518"),
        ("zero_init_mamba", 0.5, "#54A24B"),
    ]

    for key, y, color in rows:
        exp = summary["experiments"][key]
        start = datetime.fromisoformat(exp["start_utc"])
        end = datetime.fromisoformat(exp["end_utc"])
        ax.barh(y, mdates.date2num(end) - mdates.date2num(start), left=mdates.date2num(start), height=0.35, color=color, alpha=0.85)
        ax.text(start, y + 0.22, exp["label"], fontsize=9, color=color)

    for event in summary["events"]:
        ts = datetime.fromisoformat(event["time_utc"])
        ax.axvline(ts, color="#888888", linewidth=1, alpha=0.35)
        ax.scatter(ts, 3.1, s=30, color="#333333")
        ax.text(ts, 3.18, event["title"], rotation=18, fontsize=8, ha="left", va="bottom")

    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Zero-init Mamba", "Strong baseline", "Original recipe"])
    ax.set_title("Reproduction and acceptance timeline (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_ylim(0.0, 3.45)
    _savefig(FIG_DIR / "reproduction_timeline.png")


def draw_box(ax, xy, width, height, text, facecolor, fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#2D2D2D",
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)
    return patch


def draw_arrow(ax, start, end, linestyle="-", connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#404040",
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )


def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.20, 0.94, "Original CircuitFormer", fontsize=13, weight="bold", ha="center")
    ax.text(0.50, 0.945, "Architecture Comparison", fontsize=15, weight="bold", ha="center")
    ax.text(0.80, 0.94, "Accepted Variant: BEV Mamba + zero-init", fontsize=13, weight="bold", ha="center")

    ax.text(0.05, 0.86, "Original pipeline", fontsize=10.5, weight="bold", ha="left", color="#2F4B7C")
    left_boxes = [
        ((0.05, 0.70), 0.16, 0.11, "Netlist boxes\n(x1, y1, x2, y2)", "#DCEAF7"),
        ((0.30, 0.70), 0.18, 0.11, "VoxSeT encoder\npoint -> BEV", "#F7E1D7"),
        ((0.57, 0.70), 0.20, 0.11, "U-Net++ decoder\ncongestion map", "#DFF2DF"),
    ]

    ax.text(0.05, 0.56, "Accepted pipeline", fontsize=10.5, weight="bold", ha="left", color="#2C6E49")
    right_boxes = [
        ((0.05, 0.40), 0.16, 0.11, "Netlist boxes\n(x1, y1, x2, y2)", "#DCEAF7"),
        ((0.26, 0.40), 0.18, 0.11, "VoxSeT encoder\npoint -> BEV", "#F7E1D7"),
        ((0.50, 0.40), 0.18, 0.11, "BEV Mamba neck\nresidual adapter", "#FFF2B3"),
        ((0.75, 0.40), 0.20, 0.11, "U-Net++ decoder\ncongestion map", "#DFF2DF"),
    ]

    for args in left_boxes:
        draw_box(ax, args[0], args[1], args[2], args[3], args[4])
    for args in right_boxes:
        draw_box(ax, args[0], args[1], args[2], args[3], args[4])

    draw_arrow(ax, (0.21, 0.755), (0.30, 0.755))
    draw_arrow(ax, (0.48, 0.755), (0.57, 0.755))

    draw_arrow(ax, (0.21, 0.455), (0.26, 0.455))
    draw_arrow(ax, (0.44, 0.455), (0.50, 0.455))
    draw_arrow(ax, (0.68, 0.455), (0.75, 0.455))

    detail_panel = FancyBboxPatch(
        (0.06, 0.07),
        0.88,
        0.23,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.1,
        edgecolor="#9A8F4C",
        facecolor="#FFF9DD",
        zorder=0,
    )
    ax.add_patch(detail_panel)
    ax.text(0.09, 0.28, "Inside the BEV Mamba neck", fontsize=10.5, weight="bold", ha="left")

    draw_box(ax, (0.10, 0.145), 0.11, 0.06, "GroupNorm", "#FFFFFF", fontsize=8.5)
    draw_box(ax, (0.25, 0.145), 0.13, 0.06, "1x1 proj\nsplit y / gate", "#FFFFFF", fontsize=8.3)
    draw_box(ax, (0.43, 0.145), 0.11, 0.06, "DWConv", "#FFFFFF", fontsize=8.5)
    draw_box(ax, (0.59, 0.185), 0.11, 0.06, "row scan\nbi-directional", "#FFFFFF", fontsize=8.1)
    draw_box(ax, (0.59, 0.105), 0.11, 0.06, "col scan\nbi-directional", "#FFFFFF", fontsize=8.1)
    draw_box(ax, (0.75, 0.145), 0.11, 0.06, "average\n+ gate", "#FFFFFF", fontsize=8.3)
    draw_box(ax, (0.88, 0.145), 0.05, 0.06, "out\nproj", "#FFFFFF", fontsize=8.0)

    draw_arrow(ax, (0.21, 0.175), (0.25, 0.175))
    draw_arrow(ax, (0.38, 0.175), (0.43, 0.175))
    draw_arrow(ax, (0.54, 0.175), (0.59, 0.215))
    draw_arrow(ax, (0.54, 0.175), (0.59, 0.135))
    draw_arrow(ax, (0.70, 0.215), (0.75, 0.175))
    draw_arrow(ax, (0.70, 0.135), (0.75, 0.175))
    draw_arrow(ax, (0.86, 0.175), (0.88, 0.175))

    ax.text(
        0.905,
        0.115,
        "zero-init:\nstart from\nresidual-only",
        fontsize=8.0,
        ha="center",
        va="top",
        color="#6A5A00",
    )

    draw_arrow(ax, (0.59, 0.40), (0.50, 0.28), linestyle="--", connectionstyle="arc3,rad=0.15")

    ax.text(
        0.50,
        0.03,
        "Key difference: the accepted variant keeps the original encoder-decoder backbone, "
        "and adds one lightweight residual neck for row-wise and column-wise BEV scanning.",
        fontsize=9,
        ha="center",
    )

    _savefig(FIG_DIR / "architecture_diagram.png")


def make_markdown(summary):
    zero_val = summary["experiments"]["zero_init_mamba"]["final_val"]
    rerun_val = summary["experiments"]["rerun"]["final_val"]
    zero_test = summary["experiments"]["zero_init_mamba"]["test"]
    rerun_test = summary["experiments"]["rerun"]["test"]
    formal_val = summary["experiments"]["formal"]["final_val"]
    formal_test = summary["experiments"]["formal"]["test"]
    param_overhead = summary["param_overhead_vs_baseline"]

    report = f"""# CircuitFormer-Mamba 项目验收与收尾报告

## 0. Checklist

- 验收前置清单已先行编写并冻结，再进入正式验收。
- 最终验收采用的模型:
  - `exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC/epoch=99-pearson=0.6499.ckpt`
- 主要对比对象:
  - `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC/epoch=99-pearson=0.6488.ckpt`
  - `exp/congestion_formal_2026-04-11_18-09-54_UTC/epoch=99-pearson=0.5570.ckpt`

## 1. 结论摘要

当前项目的定位可概括为: **高质量复现 + 工程修复 + 一个 Mamba-inspired 轻量 neck 的探索性改进**。当前材料已经支撑以下表述: 原版 `CircuitFormer` 训练流程已打通；通过训练配方调整，强基线从 `0.5570` 提升至 `0.6488`；在强基线之上加入一个受 Mamba 思想启发的轻量二维 BEV residual neck 后，验证集与测试集上均观察到小幅正增益；参数增量约为 `+0.125%`。结论范围限定在 congestion prediction、weighted MSE 主线、单次强基线与单次 `zero-init` run。

### 1.1 术语说明: `zero-init BEV Mamba`

`zero-init BEV Mamba` 是当前验收版本的简称，这一名称由三部分组成:

- `BEV`
  表示 *Bird's-Eye View*，即俯视平面表示。在本项目中，布局点集合会先被整理为二维 BEV 特征图。

- `Mamba`
  在本项目中，`Mamba` 指向一个受 Mamba / selective state space 思想启发的二维轻量模块。模块位于 `model/bev_mamba.py`，作用范围集中在 encoder 与 decoder 之间的 neck 位置，主要由行扫描、列扫描、输入相关状态更新、门控和残差连接构成。

- `zero-init`
  表示该模块最后一层输出投影 `out_proj` 在初始化时被置零，对应代码开关为 `out_proj_init_zero=True`。

在当前项目里，`zero-init BEV Mamba` 更准确的含义可以写成:

“在 BEV 特征图上加入一个受 Mamba 思想启发的轻量残差扫描模块，并将该模块最后一层输出投影以零权重初始化。”

对应到当前验收配置，至少包含以下条件:

- `model.bev_mamba.enabled=True`
- `model.bev_mamba.num_blocks=1`
- `model.bev_mamba.inner_dim=64`
- `model.bev_mamba.scan_downsample=4`
- `model.bev_mamba.dw_kernel_size=3`
- `model.bev_mamba.out_proj_init_zero=True`

## 2. 任务定义

依据原论文 *Circuit as Set of Points* 的摘要、引言与 Method 部分，CircuitFormer 面向的是**布局后快速可布线性评估**问题。论文给出的直接任务范围包括两项:

- congestion prediction
- design rule check (DRC) violation prediction

原论文对任务形式的核心界定可以概括为: 基于电路设计的点级信息，执行网格级预测。对应英文原文短语如下:

> "making grid-level predictions of the circuit based on the point-wise information"

与输入抽象相关的原文短语如下:

> "treating circuit components as point clouds"

Method 部分进一步说明，论文将 buffers、inverters、registers、IP cores 等电路元件视作点云样本，并以每个元件的中心坐标、宽度、高度作为基础几何属性。原论文在这一处的表述同时覆盖 geometric information 与 topological information 两类来源；当前仓库主线实现直接使用的是放置后的几何信息，再通过编码器和解码器完成后续预测。

据此，原始 CircuitFormer 的任务定义可以严格表述为:

- 输入: 布局后电路元件的点级表示，重点包括几何属性，并在论文叙事中同时考虑拓扑关系
- 输出: 电路版图上的网格级预测结果
- 下游任务: congestion prediction 与 DRC violation prediction

结合当前项目的公开验收范围，本报告对任务的落点进一步限定为:

- 当前收尾结论仅覆盖 congestion prediction
- 当前代码主线从矩形框坐标出发构造几何特征，并输出 $256 \times 256$ 网格上的拥塞预测图
- 指标、图表与结项判断均围绕 congestion prediction 路线展开

## 3. 原版 CircuitFormer 的工作流程

### 3.1 输入表示

在 `model/circuitformer.py` 中，每个矩形框 $(x_1, y_1, x_2, y_2)$ 会被转换为更适合学习的几何特征:

- 中心点坐标
- 左下/右上坐标
- 宽和高
- 面积

这一表示方式同时保留了位置、尺度与形状信息。

### 3.2 编码器: VoxSeT

`model/voxelset/voxset.py` 中的 `VoxSeT` 主要包含两项关键处理:

1. 将点特征映射到隐藏空间，并叠加基于位置的傅里叶编码。
2. 在 $1\times$ / $2\times$ / $4\times$ / $8\times$ 四个尺度上执行聚合，使局部与更大范围的信息共同进入表征。

聚合后的特征会被 scatter 到 $256 \times 256$ 的 BEV 平面，后续解码器即可按照二维特征图的方式继续处理。

### 3.3 解码器: U-Net++

`model/circuitformer.py` 采用 `segmentation_models_pytorch` 中的 `UnetPlusPlus` 作为解码器，用于对整理后的二维特征图进行细化，并输出单通道拥塞图。当前实现中，解码器的 encoder 部分还会加载 `ckpts/resnet18.pth` 提供的 `resnet18` 预训练权重，因此本项目当前路线包含外部视觉预训练先验。

### 3.4 训练目标

`model/model_interface.py` 中当前验收训练主线使用的是**带像素权重的 MSE**:

- 主体误差: $(\mathrm{output}-\mathrm{label})^2$
- 再乘以数据集预先统计得到的 `weight`
- 最后乘以全局 `loss_weight=128`

`data/circuitnet.py` 中的 `weight` 来源于训练集标签分布的桶统计，并经过平滑处理。该设计使罕见但重要的拥塞区域在训练阶段获得更高关注。

需要额外说明的是，配置文件中保留了 `model.loss` 字段，当前 `training_step` 对应的正式验收训练路径为带权 MSE。报告对训练目标的表述范围据此限定在现有主线。

### 3.5 指标计算方式

`metrics.py` 对每个样本分别计算 Pearson / Spearman / Kendall，随后对样本级指标取平均。最终分数因此更接近“单张设计图平均预测质量”，较少受到少数超大样本的主导。

因此，报告中的 `test Pearson` 对应样本级相关系数均值口径，区别于将整个测试集所有像素摊平后计算单次全局相关系数。另外，当前验收配置采用单卡路径，相关说明范围限定在该执行路径。

## 4. BEV Mamba 模块原理

验收版本在 `model/circuitformer.py` 中引入了新的 `BEV Mamba neck`，结构链路为:

`VoxSeT encoder -> BEV Mamba neck -> U-Net++ decoder`

该模块位于编码器与解码器之间，输入与输出均为二维 BEV 特征图。依据现有代码，其功能可表述为: 在保持主干框架的前提下，为 BEV 特征图加入一条轻量级的行列扫描残差通道，用于补充较大范围的上下文传播。

### 4.1 代码层面的数据流

`model/bev_mamba.py` 中的 `LightweightBEVMambaBlock` 按如下顺序处理输入特征 `x`:

1. 保存残差 `residual = x`
2. 执行 `GroupNorm`
3. 若 `scan_downsample > 1`，先做平均池化降采样
4. 经过 `1x1 conv`，将通道拆成两部分:
   - 一部分作为待扫描的内容特征 `y`
   - 一部分作为门控特征 `gate`
5. 对 `y` 执行深度可分离卷积 `dwconv`
6. 分别执行行扫描与列扫描
7. 对行扫描与列扫描结果取平均
8. 通过 `sigmoid(gate)` 对结果进行门控
9. 经 `out_proj` 投影回原始通道数
10. 若前面做过降采样，则插值恢复到原分辨率
11. 与残差执行相加，输出 `residual + y`

从结构角度看，该模块同时包含三种信息处理机制:

- `dwconv` 负责局部邻域混合
- 行列扫描负责较长距离传播
- 残差连接负责尽量保持主干表征稳定，并降低新增分支的直接改写幅度

若当前 Markdown 环境支持 Mermaid，可直接渲染以下简化流程图:

```mermaid
flowchart LR
    A[布局矩形框坐标] --> B[几何特征构造]
    B --> C[VoxSeT encoder]
    C --> D[BEV feature map 256x256]
    D --> M1
    D --> R[Residual branch]

    subgraph M[BEV Mamba neck]
        M1[GroupNorm]
        M2[AvgPool if scan_downsample > 1]
        M3[1x1 Conv split to y and gate]
        M4[Depthwise Conv on y]
        M5[Row scan forward and backward]
        M6[Col scan forward and backward]
        M7[Average scans and apply gate]
        M8[Out proj]
        M1 --> M2 --> M3 --> M4
        M4 --> M5
        M4 --> M6
        M5 --> M7
        M6 --> M7
        M3 --> M7
        M7 --> M8
    end

    M8 --> F[Residual add]
    R --> F
    F --> G[U-Net++ decoder]
    G --> H[拥塞热力图输出]
```

## 5. `zero-init` 在训练起步阶段的意义

`model/bev_mamba.py` 中提供了 `out_proj_init_zero=True` 这一设置，对应如下代码逻辑:

- 新增分支最后一层 `out_proj` 的权重被初始化为 0

在参数初始化时，由于 `out_proj` 权重被显式置零且该层 bias 为空，Mamba 分支经 `out_proj` 后输出精确为 0，整个块的输出满足:

$$
\mathrm{output} = \mathrm{residual}
$$

这意味着模块在初始化时就是恒等映射，主干网络已有表征在起始阶段保持原状。

从优化角度看，该设计具有三点直接作用:

1. 新增模块在初始化时对主干特征的直接扰动为 0
2. 梯度可以先学习“在何处介入”，再学习“介入多少”
3. 插入式结构改造有助于减小训练起步阶段的直接扰动

因此，`zero-init` 可以理解为一种“保守启动”策略。网络在初始化时先维持原始主干行为，随后再学习残差修正量。

## 6. 复现历程与阶段性分析

### 6.1 第一阶段: 原版复现跑通

最早的完整复现实验为 `exp/congestion_formal_2026-04-11_18-09-54_UTC`。该阶段证明代码、数据与训练流程整体可运行，但最终验证集 Pearson 为 `{formal_val["pearson"]:.4f}`，说明“流程可运行”与“结果达到强水平”之间仍存在明显差距。

### 6.2 第二阶段: 强基线重跑

随后采用 `lr=1e-3`、`seed=3407` 进行重跑，实验目录为 `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC`。该阶段将验证集 Pearson 提升至 `{rerun_val["pearson"]:.4f}`，表明基础训练配方本身具备较大优化空间。

该阶段的意义在于建立强基线。只有在强基线存在的前提下，后续结构改动带来的增益才能获得更清晰的归因。

### 6.3 第三阶段: Mamba 改造与首次失败记录

首次 BEV Mamba 尝试为 `exp/congestion_bev_mamba_run1_2026-04-13_09-58-40_UTC`。日志中明确记录了 `FileNotFoundError`。问题来源于数据路径指向 `../datasets/...` 下一个缺失的 `.npy` 文件。

该阶段暴露出两点工程问题:

- Hydra 运行时工作目录会变化，路径配置需要显式核对。
- 训练失败分析应优先检查输入链路与数据可达性，再讨论结构层面的有效性。

### 6.4 第四阶段: `zero-init Mamba` 收口

最终接受版本为 `exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC`。训练先运行至 `epoch 89`，保存 `epoch=89-pearson=0.6475.ckpt`；中途发生中断后，从 `last.ckpt` 恢复，最终补齐至 `epoch 99`，生成 `epoch=99-pearson=0.6499.ckpt`。

该阶段表明工程流程已具备**中断恢复并保持结果连续性**的能力。同时，zero-init 实验早期标准输出的保留情况存在缺口，当前可直接核验的中间点主要包括 `epoch 89` checkpoint 与恢复后的日志结果。

## 7. 核心结果

### 7.1 最终分数总览

| 方案 | Val Pearson | Val Spearman | Val Kendall | Test Pearson | Test Spearman | Test Kendall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版首个完整复现 | {formal_val["pearson"]:.4f} | {formal_val["spearman"]:.4f} | {formal_val["kendall"]:.4f} | {formal_test["pearson"]:.4f} | {formal_test["spearman"]:.4f} | {formal_test["kendall"]:.4f} |
| 强基线重跑 | {rerun_val["pearson"]:.4f} | {rerun_val["spearman"]:.4f} | {rerun_val["kendall"]:.4f} | {rerun_test["pearson"]:.4f} | {rerun_test["spearman"]:.4f} | {rerun_test["kendall"]:.4f} |
| zero-init BEV Mamba | {zero_val["pearson"]:.4f} | {zero_val["spearman"]:.4f} | {zero_val["kendall"]:.4f} | {zero_test["pearson"]:.4f} | {zero_test["spearman"]:.4f} | {zero_test["kendall"]:.4f} |

### 7.2 Mamba 带来的增益

更合适的比较对象为**使用相同 `lr=1e-3` 与 `seed=3407` 的强基线**。在这一比较设置下，主要差异集中于结构改动本身。

相较于强基线，`zero-init BEV Mamba` 的提升如下:

- 验证集 Pearson: {summary["gain_vs_rerun"]["val"]["pearson"]["absolute"]:+.4f}
- 测试集 Pearson: {summary["gain_vs_rerun"]["test"]["pearson"]["absolute"]:+.4f}
- 测试集 Spearman: {summary["gain_vs_rerun"]["test"]["spearman"]["absolute"]:+.4f}
- 测试集 Kendall: {summary["gain_vs_rerun"]["test"]["kendall"]["absolute"]:+.4f}

需要单独指出的是，本项目从 `0.5570` 到 `0.6488` 的主要跃迁发生在强基线重跑阶段，因此当前模块的贡献可表述为“在强基线上观察到的小幅正增益”。

三个测试指标均出现同步提升，其中 Spearman 与 Kendall 的增幅高于 Pearson。基于现有指标，可以给出如下推断: 新模块带来的改进既体现在绝对值接近程度上，也体现在拥塞图高低关系的排序一致性上。该解释属于依据现有数据进行的推断，与现有结果相符；当前材料尚未给出多 seed 方差，相关结论范围据此限定在当前单次比较结果。

### 7.3 参数开销分析

参数开销较小。

基线总参数量为 `{summary["params"]["baseline"]["total"]:,}`，`zero-init Mamba` 为 `{summary["params"]["mamba_zero_init"]["total"]:,}`。新增参数量为 `{param_overhead["absolute"]:,}`，约 `+{param_overhead["relative_percent"]:.3f}%`。

这一结果表明，当前观察到的小幅增益与大幅模型扩容无关。

## 8. 图表

### 8.1 验证集 Pearson 曲线

![Validation Pearson](figures/validation_pearson_curves.png)

该图主要展示两点:

- 原版首个复现到强基线之间有明显跃迁，说明训练配方本身非常重要。
- 在强基线已经很高的情况下，`zero-init BEV Mamba` 仍有小幅抬升最终点位。

### 8.2 最终验证/测试指标柱状图

![Final Metric Bars](figures/final_metric_bars.png)

该图用于展示最终模型优劣关系。验收版本在验证集与测试集两个层面均保持领先。

### 8.3 相对强基线的净增益

![Gain vs Rerun](figures/gain_vs_rerun.png)

该图突出结构改造的净贡献。重点在于，在较强 baseline 上仍能观察到小幅正增益。

### 8.4 数据规模与参数开销

![Dataset And Params](figures/dataset_and_params.png)

该图支撑两项结论:

- 数据规模足以支撑当前项目收尾所需的结果汇总。
- Mamba 改造并未依赖大幅扩容模型规模。

### 8.5 复现与验收时间线

![Reproduction Timeline](figures/reproduction_timeline.png)

该图将“原版复现 -> 强基线 -> 首次 Mamba 失败 -> zero-init Mamba -> 中断恢复 -> 最终 checkpoint”串联为完整工程时间线，适合纳入项目总结。

### 8.6 结构示意图

![Architecture Diagram](figures/architecture_diagram.png)

该图适合用于配合结构介绍:

- 上半部分对应原版 CircuitFormer
- 下半部分对应验收版本
- 关键改动位于 encoder 与 decoder 之间的轻量残差 neck

## 9. 项目定位与改进点

本项目在当前阶段可归纳为三部分工作:

1. **高质量复现。**
   原版 `CircuitFormer` 训练流程已经跑通，并形成了可核验的 baseline checkpoint 与测试结果。

2. **工程修复。**
   训练配方、数据路径、中断恢复、日志核验与验收材料已经整理完成，强基线由 `0.5570` 提升至 `0.6488`。

3. **探索性改进。**
   在强基线之上接入一个 `Mamba-inspired` 轻量 `BEV residual neck`，当前观察到小幅正增益；参数增量约 `+0.125%`，对应一个 neck 级增量模块。

## 10. 事实说明

为保证陈述可核验，保留如下事实说明:

- `zero-init` 首轮训练的早期标准输出未完整保留为标准 `train.log`，因此 `epoch 89` 之前的中间验证点与另外两条实验相比缺少同粒度的完整可视化。报告中关于该实验的最终结论使用**已保存 checkpoint、恢复日志与重新跑出的测试日志**作为依据，未引入猜测性补点。
- 当前公开结论对应 congestion prediction 路线。配置文件中保留了 DRC label 路径入口；当前样本权重来源于 congestion 统计，因此 DRC 相关表述范围限定在代码入口说明。
- 当前训练主线使用的是带像素权重的 MSE。配置中保留了 `model.loss` 字段，正式表述范围限定在现有训练主线。
- 当前解码器的 encoder 部分加载了 `resnet18` 预训练权重，因此本报告将该路线视为包含外部视觉预训练先验。
- 当前 `BEV Mamba` 相对强基线的结论基于单次强基线与单次 zero-init run，对应“观察到的小幅正增益”；结论范围限定在当前单次比较证据。
- 项目当前定位为高质量复现、工程修复与探索性改进的组合性收尾；相关表述据此围绕已完成的复现、已完成的修复与已观察到的小幅增益展开。

## 11. 结项判断

从当前证据看，项目已经具备较完整的收尾条件:

- 原版模型已经复现
- 强基线已经建立并完成工程修复
- `Mamba-inspired` 轻量 neck 已经落地到代码与真实 checkpoint
- 中断恢复链路已经验证
- 最终验收模型在验证集和测试集上相对强基线均观察到小幅正增益
- 该增益未依赖大幅加参数

建议将 `zero-init BEV Mamba` 版本作为当前项目的归档版本，并据此整理结项文档。更严格的学术性主张仍需额外的多 seed 与范围校验材料支撑。
"""

    (OUT_DIR / "project_closeout_report.md").write_text(report)


def main():
    ensure_dirs()
    summary = build_summary()
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    plot_validation_curves(summary)
    plot_metric_bars(summary)
    plot_gain_vs_rerun(summary)
    plot_dataset_and_params(summary)
    plot_timeline(summary)
    plot_architecture_diagram()
    make_markdown(summary)

    print(f"Wrote summary to {OUT_DIR / 'summary.json'}")
    print(f"Wrote report to {OUT_DIR / 'project_closeout_report.md'}")


if __name__ == "__main__":
    main()
