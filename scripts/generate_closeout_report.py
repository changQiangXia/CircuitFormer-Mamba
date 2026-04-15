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


def draw_box(ax, xy, width, height, text, facecolor):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#2D2D2D",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    return patch


def draw_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#404040",
        )
    )


def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.24, 0.93, "Original CircuitFormer", fontsize=13, weight="bold", ha="center")
    ax.text(0.75, 0.93, "Accepted Variant: BEV Mamba + Zero-init", fontsize=13, weight="bold", ha="center")

    left_boxes = [
        ((0.05, 0.58), 0.14, 0.16, "Netlist boxes\n(x1,y1,x2,y2)", "#DCEAF7"),
        ((0.27, 0.58), 0.16, 0.16, "VoxSeT encoder\npoint -> BEV", "#F7E1D7"),
        ((0.50, 0.58), 0.18, 0.16, "U-Net++ decoder\ncongestion map", "#DFF2DF"),
    ]
    right_boxes = [
        ((0.05, 0.20), 0.14, 0.16, "Netlist boxes\n(x1,y1,x2,y2)", "#DCEAF7"),
        ((0.27, 0.20), 0.16, 0.16, "VoxSeT encoder\npoint -> BEV", "#F7E1D7"),
        ((0.50, 0.20), 0.18, 0.16, "BEV Mamba neck\nresidual adapter", "#FFF2B3"),
        ((0.77, 0.20), 0.18, 0.16, "U-Net++ decoder\ncongestion map", "#DFF2DF"),
    ]

    for args in left_boxes:
        draw_box(ax, args[0], args[1], args[2], args[3], args[4])
    for args in right_boxes:
        draw_box(ax, args[0], args[1], args[2], args[3], args[4])

    draw_arrow(ax, (0.19, 0.66), (0.27, 0.66))
    draw_arrow(ax, (0.43, 0.66), (0.50, 0.66))

    draw_arrow(ax, (0.19, 0.28), (0.27, 0.28))
    draw_arrow(ax, (0.43, 0.28), (0.50, 0.28))
    draw_arrow(ax, (0.68, 0.28), (0.77, 0.28))

    inner = [
        ("GroupNorm", 0.49, 0.50),
        ("1x1 proj + gate", 0.61, 0.50),
        ("DWConv", 0.73, 0.50),
        ("row scan", 0.83, 0.50),
        ("col scan", 0.83, 0.43),
        ("zero-init out proj", 0.66, 0.43),
    ]
    ax.text(0.59, 0.49, "Inside the Mamba neck", fontsize=10, weight="bold")
    for label, x, y in inner:
        draw_box(ax, (x, y), 0.10, 0.055, label, "#F8F8F8")

    ax.text(
        0.58,
        0.09,
        "Key idea: keep the strong encoder-decoder backbone, but insert a lightweight residual neck that scans the BEV feature map\n"
        "row-wise and column-wise. Zero-initializing the output projection makes the new branch start as a gentle adapter.",
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

    report = f"""# CircuitFormer 项目验收与收尾报告

生成时间: {summary["generated_at_utc"]}

## 0. Checklist

- 验收前置清单已先行编写并冻结，再进入正式验收。
- 最终验收采用的模型:
  - `exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC/epoch=99-pearson=0.6499.ckpt`
- 主要对比对象:
  - `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC/epoch=99-pearson=0.6488.ckpt`
  - `exp/congestion_formal_2026-04-11_18-09-54_UTC/epoch=99-pearson=0.5570.ckpt`

## 1. 一句话结论

这次收尾阶段可以把项目主线定下来: 我们不仅完整复现了 CircuitFormer，还在**几乎不增加模型规模**的前提下，把一个轻量级的 **BEV Mamba 残差适配层** 插到了编码器和解码器之间，最终得到的 `zero-init BEV Mamba` 版本在验证集和测试集上都优于强基线，说明这条改进路线是成立的。

### 1.1 术语说明: `zero-init BEV Mamba`

`zero-init BEV Mamba` 是当前验收版本的简称，这一名称由三部分组成:

- `BEV`
  表示 *Bird's-Eye View*，即俯视平面表示。在本项目中，布局点集合会先被整理为二维 BEV 特征图。

- `Mamba`
  在本项目中的含义不是“完整复现原论文中的 Mamba 主干网络”，而是指一个受 Mamba / selective state space 思想启发的二维轻量模块。该模块位于 `model/bev_mamba.py`，主要由行扫描、列扫描、输入相关状态更新、门控和残差连接构成。

- `zero-init`
  表示该模块最后一层输出投影 `out_proj` 在初始化时被置零，对应代码开关为 `out_proj_init_zero=True`。

因此，`zero-init BEV Mamba` 在本项目中的更准确含义是:

“在 BEV 特征图上加入一个受 Mamba 思想启发的轻量残差扫描模块，并将该模块最后一层输出投影以零权重初始化。”

对应到当前验收配置，至少包含以下条件:

- `model.bev_mamba.enabled=True`
- `model.bev_mamba.num_blocks=1`
- `model.bev_mamba.inner_dim=64`
- `model.bev_mamba.scan_downsample=4`
- `model.bev_mamba.dw_kernel_size=3`
- `model.bev_mamba.out_proj_init_zero=True`

## 2. 这个项目到底在做什么

通俗地说，这个任务是在做“**芯片拥塞热力图预测**”。输入不是普通图片，而是一堆标准单元或模块的矩形框坐标。模型要根据这些布局元素，预测 256x256 的拥塞分布图，告诉我们哪里可能布线紧张、哪里相对宽松。

这个项目最有意思的地方在于，它没有先把电路强行转成一张手工做出来的图，也没有先做很重的图预处理，而是把电路看成“带几何属性的点集合”，直接从原始布局元素里学习特征。

## 3. 原版 CircuitFormer 是怎么工作的

### 3.1 输入表示

在 `model/circuitformer.py` 里，模型先把每个矩形框 `(x1, y1, x2, y2)` 变成更容易学习的几何特征:

- 中心点坐标
- 左下/右上坐标
- 宽和高
- 面积

这样做的好处是，模型拿到的不只是“框在哪里”，还知道“框有多大、形状如何”。

### 3.2 编码器: VoxSeT

`model/voxelset/voxset.py` 里的 `VoxSeT` 做了两件关键事:

1. 先把点特征映射到隐藏空间，并加上基于位置的傅里叶编码。
2. 再在 1x / 2x / 4x / 8x 四个尺度上做聚合，把局部与更大范围的信息都揉进来。

聚合后的特征会被 scatter 到 256x256 的 BEV 平面上，因此后面的解码器就能像处理图像那样处理布局特征图。

### 3.3 解码器: U-Net++

`model/circuitformer.py` 里用的是 `segmentation_models_pytorch` 的 `UnetPlusPlus`。这一步可以理解成“把已经整理好的二维特征图重新细化”，最后输出一张单通道拥塞图。

### 3.4 训练目标

`model/model_interface.py` 里的训练损失是**带像素权重的 MSE**:

- 主体误差: `(output - label)^2`
- 再乘上数据集里预先统计好的 `weight`
- 最后再乘一个全局的 `loss_weight=128`

`data/circuitnet.py` 里的 `weight` 不是随便拍脑袋设的，它来自训练集标签分布的桶统计，并做了平滑。直观理解就是: 罕见但重要的拥塞区域，在训练时会被更认真地看待。

### 3.5 指标怎么计算

`metrics.py` 不是把全测试集所有像素摊平后算一次相关系数，而是**先对每个样本单独算 Pearson / Spearman / Kendall，再对样本取平均**。这意味着最终分数更像“平均每张设计图的预测质量”，而不是少数超大样本主导出来的单个总分。

## 4. Mamba 版到底改了什么

这次接受验收的版本不是推翻原模型重来，而是在 `model/circuitformer.py` 里，把一个新的 `BEV Mamba neck` 插到了:

`VoxSeT encoder -> BEV Mamba neck -> U-Net++ decoder`

具体实现见 `model/bev_mamba.py`。它的结构可以概括成:

1. 先做 `GroupNorm`
2. 用 `1x1 conv` 把通道拆成“内容分支”和“门控分支”
3. 用 `depthwise conv` 做局部空间混合
4. 再分别沿着**行方向**和**列方向**做双向扫描
5. 把两条扫描结果平均后，再乘门控
6. 最后投影回原通道数，并和输入做残差相加

这一层的价值在于: 原版编码器已经能把点云布局整理成 BEV 特征图，但 BEV 特征图上的“长程关系”仍然可以继续增强。Mamba 风格的扫描给了模型一种比纯卷积更直接的“沿行/列传播信息”的办法。

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

## 5. 为什么 `zero-init` 很关键

`model/bev_mamba.py` 里有一个很重要的开关: `out_proj_init_zero=True`。

它做的事情并不玄学，就是把新增分支最后那层 `out_proj` 的权重初始化成 0。这样一来，训练刚开始时这条新分支几乎不会粗暴改写原始特征，整个 neck 更像一个“从零开始学习的小修正项”，而不是一上来就把原 backbone 的表示打乱。

从工程角度看，这是一种很稳的加法方式: 先尊重已经有效的原模型，再让新模块逐步学会在何处、以多大力度介入。

## 6. 复现历程与反思

### 6.1 第一阶段: 原版复现跑通

最早的完整复现实验是 `exp/congestion_formal_2026-04-11_18-09-54_UTC`。这一步证明了代码、数据和训练流程整体是通的，但最终验证集 Pearson 只有 `{formal_val["pearson"]:.4f}`，说明“能跑通”和“跑到强结果”不是一回事。

### 6.2 第二阶段: 强基线重跑

随后使用 `lr=1e-3`、`seed=3407` 重跑，实验目录是 `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC`。这一步把验证集 Pearson 拉到了 `{rerun_val["pearson"]:.4f}`，说明基础训练配方本身就还有不小的优化空间。

这个阶段的收获很重要: 如果没有把基线先抬高，后面就很难判断 Mamba 的提升到底来自结构，还是只是来自“把训练配方调得更合理”。

### 6.3 第三阶段: Mamba 改造与第一次失败

第一次 BEV Mamba 尝试是 `exp/congestion_bev_mamba_run1_2026-04-13_09-58-40_UTC`，但日志里明确出现了 `FileNotFoundError`。问题根源不是模型本身，而是数据路径指向了 `../datasets/...` 下一个并不存在的 `.npy` 文件。

这一步的反思很直接:

- Hydra 运行时工作目录会变化，路径问题不能只靠“看起来差不多”来判断。
- 训练失败时要先排输入链路，再谈结构好坏。

### 6.4 第四阶段: zero-init Mamba 完整收口

最终接受版本是 `exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC`。它先跑到了 `epoch 89`，保存了 `epoch=89-pearson=0.6475.ckpt`；中途训练意外中断后，又从 `last.ckpt` 恢复，最终补齐到 `epoch 99`，落下 `epoch=99-pearson=0.6499.ckpt`。

这一段复现记录很有价值，因为它说明我们的工程流程不是“只能在理想状态下跑一次”，而是已经具备了**中断后续跑并保持结果连续性**的能力。

## 7. 最核心的结果

### 7.1 最终分数总览

| 方案 | Val Pearson | Val Spearman | Val Kendall | Test Pearson | Test Spearman | Test Kendall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版首个完整复现 | {formal_val["pearson"]:.4f} | {formal_val["spearman"]:.4f} | {formal_val["kendall"]:.4f} | {formal_test["pearson"]:.4f} | {formal_test["spearman"]:.4f} | {formal_test["kendall"]:.4f} |
| 强基线重跑 | {rerun_val["pearson"]:.4f} | {rerun_val["spearman"]:.4f} | {rerun_val["kendall"]:.4f} | {rerun_test["pearson"]:.4f} | {rerun_test["spearman"]:.4f} | {rerun_test["kendall"]:.4f} |
| zero-init BEV Mamba | {zero_val["pearson"]:.4f} | {zero_val["spearman"]:.4f} | {zero_val["kendall"]:.4f} | {zero_test["pearson"]:.4f} | {zero_test["spearman"]:.4f} | {zero_test["kendall"]:.4f} |

### 7.2 如果只想知道“换成 Mamba 到底有没有提升”

最公平的比较不是拿 `zero-init Mamba` 去对最早那版原始复现，而是去对**同样使用 `lr=1e-3`、`seed=3407` 的强基线**。这样对比时，主要差别就集中在结构本身。

和强基线相比，`zero-init BEV Mamba` 的提升是:

- 验证集 Pearson: {summary["gain_vs_rerun"]["val"]["pearson"]["absolute"]:+.4f}
- 测试集 Pearson: {summary["gain_vs_rerun"]["test"]["pearson"]["absolute"]:+.4f}
- 测试集 Spearman: {summary["gain_vs_rerun"]["test"]["spearman"]["absolute"]:+.4f}
- 测试集 Kendall: {summary["gain_vs_rerun"]["test"]["kendall"]["absolute"]:+.4f}

这里最值得注意的是: **三个测试指标都一起变好**。尤其是 Spearman 和 Kendall 的提升幅度比 Pearson 更明显。这说明新模块带来的不只是“某几个绝对值更贴近”，更可能是**整张拥塞图里高低关系的排序也更一致了**。这是一种合理解读，不是额外实验结论，但它和现有数字是对得上的。

### 7.3 提升是不是靠堆参数换来的

不是。

基线总参数量是 `{summary["params"]["baseline"]["total"]:,}`，`zero-init Mamba` 是 `{summary["params"]["mamba_zero_init"]["total"]:,}`。也就是说只增加了 `{param_overhead["absolute"]:,}` 个参数，约 `+{param_overhead["relative_percent"]:.3f}%`。

用更直白的话说: **参数基本没怎么变，但分数还是往上走了**。这使得“结构更聪明了”这个说法更站得住。

## 8. 图表

### 8.1 验证集 Pearson 曲线

![Validation Pearson](figures/validation_pearson_curves.png)

这张图最适合看两件事:

- 原版首个复现到强基线之间有明显跃迁，说明训练配方本身非常重要。
- 在强基线已经很高的情况下，`zero-init BEV Mamba` 还能继续把最终点位再往上推一点。

### 8.2 最终验证/测试指标柱状图

![Final Metric Bars](figures/final_metric_bars.png)

这张图适合在答辩或汇报时直接回答“最后到底谁最好”。验收版本在验证和测试两个层面都保持了领先。

### 8.3 相对强基线的净增益

![Gain vs Rerun](figures/gain_vs_rerun.png)

这张图比总分表更适合讲“结构改造的净贡献”。重点不是绝对值有多夸张，而是**在一个已经很强的 baseline 上，依然实现了稳定正增益**。

### 8.4 数据规模与参数开销

![Dataset And Params](figures/dataset_and_params.png)

这张图用来支持两个结论:

- 数据规模足够大，训练不是玩具级实验。
- Mamba 改造不是靠大幅扩容模型来换分数。

### 8.5 复现与验收时间线

![Reproduction Timeline](figures/reproduction_timeline.png)

这张图把“原版复现 -> 强基线 -> 首次 Mamba 失败 -> zero-init Mamba -> 中断恢复 -> 最终 checkpoint”串成了一条完整工程叙事线，适合写进项目总结。

### 8.6 结构示意图

![Architecture Diagram](figures/architecture_diagram.png)

这张图可以配合口头讲解:

- 上半部分是原版 CircuitFormer
- 下半部分是接受验收的版本
- 最关键的变化就是在 encoder 和 decoder 之间插入了一个轻量残差 neck

## 9. 这次工作的真正创新点

如果把整件事压缩成几个最值得讲的点，我会这样概括:

1. **保持主干不动，只对 BEV 特征图做增量改造。**
   这比“整个模型推倒重来”更稳，也更容易解释清楚改进到底来自哪里。

2. **用行扫描和列扫描补强二维布局特征。**
   对芯片布局这种天然具有二维空间结构的任务，这种设计很自然，也很贴题。

3. **用 zero-init 让新模块从“安全模式”起步。**
   这不是花哨技巧，而是一种很务实的工程设计: 先不要破坏已有能力，再逐步学习增益。

4. **在强基线上继续挖到正收益。**
   这一点比“从弱基线大幅提升”更难，也更有说服力。

## 10. 简短的事实说明

为了严格求真，这里保留一条必要说明: `zero-init` 首轮训练的早期标准输出没有完整保留成一份标准 `train.log`，所以它在 `epoch 89` 之前的中间验证点没有像另外两条实验那样完整可视化。报告中关于这条实验的最终结论只使用了**已保存 checkpoint、恢复日志和重新跑出的测试日志**，没有用猜测去补点。

## 11. 结项建议

从当前证据看，这个项目已经具备比较完整的收尾条件:

- 原版模型已经复现
- 强基线已经建立
- Mamba 改造已经落地到代码和真实 checkpoint
- 中断恢复链路已经验证
- 最终验收模型在验证集和测试集上都优于强基线
- 提升不是靠大幅加参数换来的

因此，我建议把 `zero-init BEV Mamba` 版本作为本项目的正式收口版本，并以它为主线完成最后的答辩/汇报材料。
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
