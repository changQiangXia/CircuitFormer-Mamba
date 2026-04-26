#!/usr/bin/env python3
import csv
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
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


def parse_history_from_metrics_csv(path):
    history = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("pearson"):
                continue
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "pearson": float(row["pearson"]),
                    "spearman": float(row["spearman"]),
                    "kendall": float(row["kendall"]),
                }
            )
    if not history:
        raise ValueError(f"No validation history found in {path}")
    return history


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


def experiment_display_label(key):
    labels = {
        "formal": "Original\nrecipe",
        "rerun": "Strong\nbaseline",
        "zero_init_mamba": "zero-init\nBEV Mamba",
        "true_mamba_scheme_b": "True Mamba\nScheme B",
    }
    return labels.get(key, key)


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
    true_metrics = REPO_ROOT / "exp" / "congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC" / "csv_logs" / "version_0" / "metrics.csv"
    true_ckpt_99 = REPO_ROOT / "exp" / "congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC" / "epoch=99-pearson=0.6467.ckpt"
    true_test = REPO_ROOT / "exp" / "congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC" / "acceptance_test_epoch99.log"

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
    true_history = parse_history_from_metrics_csv(true_metrics)
    true_final_val = true_history[-1]

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
            "true_mamba_scheme_b": {
                "label": "True Mamba neck (Scheme B, exploratory)",
                "start_utc": parse_run_start_utc("2026-04-24_02-03-01_UTC").isoformat(),
                "end_utc": file_mtime_utc(true_ckpt_99).isoformat(),
                "best_ckpt": "exp/congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC/epoch=99-pearson=0.6467.ckpt",
                "val_history": true_history,
                "final_val": true_final_val,
                "test": parse_single_metric_from_log(true_test) if true_test.exists() else None,
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
                "detail": "The resumed run reaches epoch 99 and saves the final zero-init checkpoint.",
            },
            {
                "time_utc": parse_run_start_utc("2026-04-24_02-03-01_UTC").isoformat(),
                "title": "True Mamba Scheme B starts",
                "detail": "A separate exploratory line starts from the same congestion recipe and inserts an official Mamba-based neck.",
            },
            {
                "time_utc": file_mtime_utc(true_ckpt_99).isoformat(),
                "title": "True Mamba final checkpoint saved",
                "detail": "The exploratory Scheme B run reaches epoch 99 and saves a validation-best checkpoint.",
            },
        ],
        "notes": {
            "metric_definition": "Pearson, Spearman, and Kendall are computed per sample on the predicted and target congestion maps, then averaged across the dataset.",
            "validation_cadence": "Validation runs every 10 epochs according to trainer.check_val_every_n_epoch=10.",
            "zero_init_history_note": "The early stdout of the zero-init run was not preserved as a full train.log, so its intermediate curve before epoch 89 is reconstructed only from saved checkpoints and the later resume log.",
            "run1_failure_excerpt": "FileNotFoundError: [Errno 2] No such file or directory: '../datasets/CircuitNet-N28/graph_features/instance_placement_micron/6274-RISCY-FPU-b-1-c20-u0.9-m1-p4-f1.npy'",
            "project_position_note": "This project is positioned as a high-quality reproduction, engineering repair, and parallel comparison of two Mamba-related neck routes.",
            "method_scope_note": "The lightweight BEV Mamba path is a Mamba-inspired residual scan neck, while the true-Mamba path wraps the official mamba_ssm.Mamba module at the same neck insertion point.",
            "contribution_scope_note": "The largest Pearson jump in this project comes from the strong baseline rerun; the two Mamba routes are recorded as route-specific observations on top of that baseline.",
            "experiment_scope_note": "Public conclusions here cover congestion prediction under the current weighted-MSE training path, using one strong-baseline run, one zero-init BEV Mamba run, and one true Mamba Scheme B run. The DRC entry in config and the configurable loss field remain code-level interfaces rather than finalized experimental tracks.",
            "prior_note": "The decoder encoder loads ResNet18 pretrained weights, so the current routes include an external visual pretraining prior.",
            "variance_note": "Each comparison route in the current archive is represented by a single run.",
            "true_mamba_note": "The true Mamba Scheme B line uses official mamba_ssm.Mamba modules under a separately validated environment and includes both validation history and test log.",
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

    variant_gains_vs_rerun = {}
    for variant_key in ("zero_init_mamba", "true_mamba_scheme_b"):
        variant_gains_vs_rerun[variant_key] = {}
        variant_val = summary["experiments"][variant_key]["final_val"]
        variant_gains_vs_rerun[variant_key]["val"] = {}
        for key in ("pearson", "spearman", "kendall"):
            abs_gain = variant_val[key] - rerun_val[key]
            rel_gain = 100.0 * abs_gain / rerun_val[key]
            variant_gains_vs_rerun[variant_key]["val"][key] = {
                "absolute": abs_gain,
                "relative_percent": rel_gain,
            }
        variant_test = summary["experiments"][variant_key]["test"]
        if variant_test:
            variant_gains_vs_rerun[variant_key]["test"] = {}
            for key in ("pearson", "spearman", "kendall"):
                abs_gain = variant_test[key] - rerun_test_metrics[key]
                rel_gain = 100.0 * abs_gain / rerun_test_metrics[key]
                variant_gains_vs_rerun[variant_key]["test"][key] = {
                    "absolute": abs_gain,
                    "relative_percent": rel_gain,
                }
    summary["variant_gains_vs_rerun"] = variant_gains_vs_rerun

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
        "true_mamba_scheme_b": "#B279A2",
    }
    linestyles = {
        "formal": "-",
        "rerun": "-",
        "zero_init_mamba": "--",
        "true_mamba_scheme_b": "-.",
    }
    text_offsets = {
        "formal": (0.8, 0.0000),
        "rerun": (0.8, -0.0018),
        "zero_init_mamba": (0.8, 0.0018),
        "true_mamba_scheme_b": (0.8, -0.0055),
    }
    for key in ("formal", "rerun", "zero_init_mamba", "true_mamba_scheme_b"):
        history = summary["experiments"][key]["val_history"]
        xs = [item["epoch"] for item in history]
        ys = [item["pearson"] for item in history]
        plt.plot(xs, ys, marker="o", linewidth=2.2, linestyle=linestyles[key], color=colors[key], label=summary["experiments"][key]["label"])
        plt.scatter(xs[-1], ys[-1], s=80, color=colors[key])
        dx, dy = text_offsets[key]
        plt.text(xs[-1] + dx, ys[-1] + dy, f"{ys[-1]:.4f}", fontsize=9, color=colors[key])

    plt.xlabel("Epoch")
    plt.ylabel("Validation Pearson")
    plt.title("Validation Pearson Across Four Reproduced / Exploratory Runs")
    plt.grid(alpha=0.25, linestyle="--")
    plt.xticks([9, 19, 29, 39, 49, 59, 69, 79, 89, 99])
    plt.legend(frameon=False)
    _savefig(FIG_DIR / "validation_pearson_curves.png")


def plot_metric_bars(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharey=False)
    exp_keys = ["formal", "rerun", "zero_init_mamba", "true_mamba_scheme_b"]
    metrics = ["pearson", "spearman", "kendall"]
    x = list(range(len(exp_keys)))
    width = 0.22
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for ax, split_name, getter in [
        (axes[0], "Validation", lambda key: summary["experiments"][key]["final_val"]),
        (axes[1], "Test", lambda key: summary["experiments"][key]["test"]),
    ]:
        missing_groups = set()
        for idx, metric in enumerate(metrics):
            positions = []
            values = []
            for group_idx, key in enumerate(exp_keys):
                metric_block = getter(key)
                xpos = group_idx + (idx - 1) * width
                if metric_block is None or metric not in metric_block:
                    missing_groups.add(group_idx)
                    continue
                positions.append(xpos)
                values.append(metric_block[metric])
            ax.bar(positions, values, width=width, label=metric.capitalize(), color=colors[idx])
            for xp, yp in zip(positions, values):
                ax.text(xp, yp + 0.003, f"{yp:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([experiment_display_label(key) for key in exp_keys], rotation=0, ha="center")
        ax.set_title(f"{split_name} Metrics")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_ylim(0.30, 0.725)
        for group_idx in sorted(missing_groups):
            left = group_idx - 1.55 * width
            ax.add_patch(
                Rectangle(
                    (left, 0.302),
                    3.1 * width,
                    0.013,
                    facecolor="#EFEFEF",
                    edgecolor="#B8B8B8",
                    hatch="//",
                    linewidth=0.8,
                    zorder=0,
                )
            )
            ax.text(group_idx, 0.318, "N/A", ha="center", va="bottom", fontsize=8.5, color="#666666")
        if split_name == "Test" and missing_groups:
            ax.text(2.55, 0.338, "True Mamba test log pending", fontsize=8.5, color="#666666")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Final Validation/Test Metrics Across Four Runs", y=0.99)
    fig.subplots_adjust(top=0.76)
    _savefig(FIG_DIR / "final_metric_bars.png")


def plot_gain_vs_rerun(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    metrics = ["pearson", "spearman", "kendall"]
    variant_keys = ["zero_init_mamba", "true_mamba_scheme_b"]
    variant_colors = {
        "zero_init_mamba": "#54A24B",
        "true_mamba_scheme_b": "#B279A2",
    }

    for ax, split_name in zip(axes, ["val", "test"]):
        available = [key for key in variant_keys if split_name in summary["variant_gains_vs_rerun"][key]]
        width = 0.34 if len(available) > 1 else 0.54
        x = list(range(len(metrics)))
        all_values = []
        for idx, variant_key in enumerate(available):
            gains = [summary["variant_gains_vs_rerun"][variant_key][split_name][metric]["absolute"] for metric in metrics]
            offset = (idx - (len(available) - 1) / 2) * width
            xpos = [value + offset for value in x]
            bars = ax.bar(
                xpos,
                gains,
                width=width,
                color=variant_colors[variant_key],
                label=summary["experiments"][variant_key]["label"],
            )
            all_values.extend(gains)
            for bar, value, metric in zip(bars, gains, metrics):
                rel = summary["variant_gains_vs_rerun"][variant_key][split_name][metric]["relative_percent"]
                label = f"{value:+.4f}\n({rel:+.2f}%)"
                if value >= 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.0006,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )
                else:
                    ax.annotate(
                        f"{value:+.4f}\n({rel:+.2f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, value * 0.5),
                        xytext=(18, -2),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=7.0,
                        color="#222222",
                    )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"{split_name.capitalize()} gain vs strong baseline")
        ax.set_ylabel("Absolute metric gain")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([metric.capitalize() for metric in metrics])
        ax.tick_params(axis="x", pad=12)
        ymin = min(all_values + [0.0])
        ymax = max(all_values + [0.0])
        lower = min(ymin * 2.6, -0.0068) if ymin < 0 else -0.004
        upper = ymax * 1.55 if ymax > 0 else 0.01
        ax.set_ylim(lower, upper)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Increment over the strong baseline", y=1.08)
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
    axes[1].bar(["Baseline", "zero-init\nBEV Mamba"], param_values, color=["#F58518", "#54A24B"])
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
    fig, ax = plt.subplots(figsize=(12.2, 5.3))

    rows = [
        ("true_mamba_scheme_b", 3.5, "#B279A2"),
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

    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(["Zero-init Mamba", "Strong baseline", "Original recipe", "True Mamba"])
    ax.set_title("Reproduction timeline (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_ylim(0.0, 4.45)
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
    ax.text(0.80, 0.94, "Lightweight BEV Mamba Route", fontsize=13, weight="bold", ha="center")

    ax.text(0.05, 0.86, "Original pipeline", fontsize=10.5, weight="bold", ha="left", color="#2F4B7C")
    left_boxes = [
        ((0.05, 0.70), 0.16, 0.11, "Netlist boxes\n(x1, y1, x2, y2)", "#DCEAF7"),
        ((0.30, 0.70), 0.18, 0.11, "VoxSeT encoder\npoint -> BEV", "#F7E1D7"),
        ((0.57, 0.70), 0.20, 0.11, "U-Net++ decoder\ncongestion map", "#DFF2DF"),
    ]

    ax.text(0.05, 0.56, "Lightweight BEV Mamba pipeline", fontsize=10.5, weight="bold", ha="left", color="#2C6E49")
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
        "Key difference: this route keeps the original encoder-decoder backbone, "
        "adds one lightweight residual neck for row-wise and column-wise BEV scanning, "
        "and shares the same insertion point with the true-Mamba route.",
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
    true_val = summary["experiments"]["true_mamba_scheme_b"]["final_val"]
    true_test = summary["experiments"]["true_mamba_scheme_b"]["test"]
    zero_gain = summary["variant_gains_vs_rerun"]["zero_init_mamba"]
    true_gain = summary["variant_gains_vs_rerun"]["true_mamba_scheme_b"]
    param_overhead = summary["param_overhead_vs_baseline"]

    report = f"""# CircuitFormer-Mamba 项目记录与结果报告

## 0. Checklist

- 前置 checklist 已冻结，实验推进与文档整理均据此执行。
- 四条已归档实验线:
  - `exp/congestion_formal_2026-04-11_18-09-54_UTC/epoch=99-pearson=0.5570.ckpt`
  - `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC/epoch=99-pearson=0.6488.ckpt`
  - `exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC/epoch=99-pearson=0.6499.ckpt`
  - `exp/congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC/epoch=99-pearson=0.6467.ckpt`
- 当前公开对照基线:
  - `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC/epoch=99-pearson=0.6488.ckpt`

## 1. 定位与范围

当前项目的定位可概括为: **高质量复现 + 工程修复 + 两条 Mamba 相关 neck 路线的并行探索与对照**。

当前材料支撑以下事实:

- 原版 `CircuitFormer` 训练流程已打通。
- 训练配方调整将验证集 Pearson 从 `0.5570` 提升至 `0.6488`。
- `zero-init BEV Mamba` 路线在强基线之上带来小幅上扬，参数增量约 `+0.125%`。
- `true Mamba Scheme B` 路线调用官方 `mamba_ssm.Mamba` 模块，在当前单次 run 中呈现“Pearson 接近强基线、Spearman 与 Kendall 高于强基线”的指标形态。

公开结论范围限定在 congestion prediction、带像素权重的 MSE 主线、单次 strong baseline、单次 `zero-init BEV Mamba`、单次 `true Mamba Scheme B`。

### 1.1 两条 Mamba 路线的名称与作用范围

- `zero-init BEV Mamba`
  - 代码位置: `model/bev_mamba.py`
  - 结构定位: encoder 与 decoder 之间的轻量二维残差 neck
  - 名称含义: `zero-init` 指 `out_proj` 以零权重初始化；`BEV Mamba` 指向受 selective state space 思路启发的行列扫描模块

- `true Mamba Scheme B`
  - 代码位置: `true_mamba_experiments/modules/true_mamba_neck.py`
  - 结构定位: 与轻量 neck 相同的插入位置
  - 名称含义: 行扫描与列扫描的核心序列混合器直接调用官方 `mamba_ssm.Mamba`

- `model/circuitformer.py` 对两条路线设置了互斥保护，同一时间只启用一个 neck 路线。

## 2. 任务定义

依据原论文 *Circuit as Set of Points* 的摘要、引言与 Method 部分，CircuitFormer 面向的是布局后快速可布线性评估。论文给出的直接任务范围包括两项:

- congestion prediction
- design rule check (DRC) violation prediction

原论文对任务形式的核心界定可概括为: 基于电路设计的点级信息，执行网格级预测。对应英文原文短语如下:

> "making grid-level predictions of the circuit based on the point-wise information"

与输入抽象相关的原文短语如下:

> "treating circuit components as point clouds"

据此，原始 CircuitFormer 的任务定义可严格表述为:

- 输入: 布局后电路元件的点级表示，重点包括几何属性，并在论文叙事中同时考虑拓扑关系
- 输出: 电路版图上的网格级预测结果
- 下游任务: congestion prediction 与 DRC violation prediction

结合当前项目的公开范围，本报告进一步限定如下:

- 当前结果围绕 congestion prediction 路线展开
- 当前代码主线从矩形框坐标出发构造几何特征，并输出 $256 \\times 256$ 网格上的拥塞预测图
- 指标、图表与路线对照均围绕 congestion prediction 展开

## 3. 原版 CircuitFormer 主线

### 3.1 输入表示

在 `model/circuitformer.py` 中，每个矩形框 $(x_1, y_1, x_2, y_2)$ 会被转换为更适合学习的几何特征:

- 中心点坐标
- 左下/右上坐标
- 宽和高
- 面积

这一表示方式同时保留了位置、尺度与形状信息。

### 3.2 编码器与解码器

`model/voxelset/voxset.py` 中的 `VoxSeT` 会将点特征映射到隐藏空间，并在 $1\\times$ / $2\\times$ / $4\\times$ / $8\\times$ 四个尺度上聚合信息。聚合后的特征被 scatter 到 $256 \\times 256$ 的 BEV 平面。

`model/circuitformer.py` 采用 `segmentation_models_pytorch` 中的 `UnetPlusPlus` 作为解码器，用于对二维 BEV 特征图进行细化，并输出单通道拥塞图。当前实现中，解码器 encoder 还会加载 `ckpts/resnet18.pth` 提供的 `resnet18` 预训练权重，因此当前路线包含外部视觉预训练先验。

### 3.3 训练目标

`model/model_interface.py` 的当前训练主线使用带像素权重的 MSE:

$$
\mathcal{{L}} = 128 \cdot w \cdot (\hat{{y}} - y)^2
$$

其中 $w$ 来自 `data/circuitnet.py` 中基于训练集标签分布构造的权重图。该设计使高拥塞区域在训练阶段获得更高关注。

配置文件中保留了 `model.loss` 字段，当前正式训练路径对应上述带权 MSE。

### 3.4 指标口径

`metrics.py` 对每个样本分别计算 Pearson / Spearman / Kendall，随后对样本级指标取平均。因此，报告中的 `test Pearson` 对应“逐样本相关系数均值”，与“全测试集像素摊平后计算单次全局相关系数”属于不同口径。

## 4. 两条 Mamba 相关 neck 路线

### 4.1 `zero-init BEV Mamba`

`model/bev_mamba.py` 中的轻量 neck 由 `GroupNorm`、`1x1` 通道投影、深度可分离卷积、行列双向扫描、门控与残差连接构成。核心状态更新写在 `SequenceStateSpace.forward` 中:

$$
s_t = (1 - \delta_t) \odot s_{{t-1}} + \delta_t \odot v_t
$$

其中

$$
\delta_t = \sigma(W_\delta x_t), \qquad v_t = W_v x_t
$$

代码层面的含义较直接:

- `delta_proj` 生成输入相关的更新系数
- `value_proj` 生成待写入状态的内容
- `for idx in range(seq.shape[1])` 显式推进时间步
- 行扫描与列扫描各自做正向、反向两次遍历，再取平均

这一实现更贴近 **Mamba-inspired lightweight scan neck** 这一表述。严格口径下，官方 `selective scan kernel` 并未出现在该文件中。

`zero-init` 的含义来自 `out_proj_init_zero=True`。在初始化时，输出投影层权重被显式置零，因此新增分支在起步阶段满足:

$$
\mathrm{{output}} = \mathrm{{residual}}
$$

该设置使主干表征在训练早期保持稳定，随后再逐步学习残差修正量。

若当前 Markdown 环境支持 Mermaid，可直接渲染以下简化流程图:

```mermaid
flowchart LR
    A[BEV feature map] --> B[GroupNorm]
    B --> C[AvgPool if needed]
    C --> D[1x1 conv split y and gate]
    D --> E[Depthwise Conv on y]
    E --> F[Row scan forward and backward]
    E --> G[Col scan forward and backward]
    F --> H[Average scans]
    G --> H
    D --> I[Sigmoid gate]
    H --> J[Apply gate]
    I --> J
    J --> K[out_proj]
    K --> L[Upsample if needed]
    L --> M[Residual add]
```

### 4.2 `true Mamba Scheme B`

`true_mamba_experiments/modules/true_mamba_neck.py` 中的 `TrueMambaBlock` 直接导入官方 `mamba_ssm.Mamba`，并将二维 BEV 特征图拆成行序列与列序列:

- 行序列形状: `[B * H, W, C]`
- 列序列形状: `[B * W, H, C]`

双向扫描的写法为:

$$
f_{{\mathrm{{bi}}}}(z) = \\frac{{1}}{{2}}\\left(\\mathrm{{Mamba}}(z) + \\mathrm{{flip}}(\\mathrm{{Mamba}}(\\mathrm{{flip}}(z)))\\right)
$$

行、列结果再取平均:

$$
y = \\frac{{1}}{{2}}\\left(f_{{\mathrm{{row}}}}(x) + f_{{\mathrm{{col}}}}(x)\\right)
$$

当前代码还叠加了若干工程保护:

- `torch.autocast(..., enabled=False)` 使状态空间扫描保持在 `fp32`
- `_ensure_finite(...)` 在 `norm`、`downsample`、`row_scan`、`col_scan`、`out_proj`、`residual_add` 等阶段检查非有限值
- `use_mask=True` 时，仅在有效占用区域上执行扫描与回写
- `downsample > 1` 时，先缩小扫描分辨率，再插值回原尺度
- `use_residual_scale=True` 时，对新增分支额外乘一个可学习缩放系数

当前归档的 Scheme B 训练脚本位于 `true_mamba_experiments/scripts/train_congestion_true_mamba_scheme_b.sh`，核心设置为:

- `downsample=4`
- `use_input_norm=False`
- `use_mask=True`
- `mask_pool_mode=max`
- `out_proj_init_zero=False`
- `out_proj_init_std=0.001`
- `use_residual_scale=True`
- `residual_scale_init=0.001`
- `trainer.precision=bf16-mixed`
- `trainer.gradient_clip_val=1.0`

### 4.3 与 Mamba 原论文的对应关系

Mamba 原论文为 *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*，论文链接为 <https://arxiv.org/abs/2312.00752>。标题中的 `Selective State Spaces` 与摘要中的 “letting the SSM parameters be functions of the input” 指向输入相关的状态空间更新。

当前项目中，两条路线与原论文的对应关系可作如下区分:

- `zero-init BEV Mamba`
  - 对应关系: 保留“输入相关更新 + 序列扫描 + 残差修正”这一思路
  - 代码边界: 手写 `SequenceStateSpace` 与显式 `for` 循环推进
  - 更贴切的术语: `Mamba-inspired lightweight neck`

- `true Mamba Scheme B`
  - 对应关系: 行扫描与列扫描的核心混合器直接调用官方 `mamba_ssm.Mamba`
  - 代码边界: 官方 selective SSM 核心来自上游库；二维展开、双向平均、mask、下采样、上采样、残差拼接属于项目侧封装
  - 更贴切的术语: `official-Mamba neck route`

## 5. 复现与工程记录

### 5.1 原版复现跑通

最早的完整复现实验为 `exp/congestion_formal_2026-04-11_18-09-54_UTC`。该阶段验证了代码、数据与训练流程整体可运行，最终验证集 Pearson 为 `{formal_val["pearson"]:.4f}`。

### 5.2 强基线重跑

随后采用 `lr=1e-3`、`seed=3407` 进行重跑，实验目录为 `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC`。该阶段将验证集 Pearson 提升至 `{rerun_val["pearson"]:.4f}`。从 `0.5570` 到 `0.6488` 的主要跃迁发生在这一阶段。

### 5.3 `zero-init BEV Mamba` 路线

对应实验目录为 `exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC`。训练先运行至 `epoch 89`，保存 `epoch=89-pearson=0.6475.ckpt`；中途发生中断后，从 `last.ckpt` 恢复，最终补齐至 `epoch 99`，生成 `epoch=99-pearson=0.6499.ckpt`。

该阶段确认了两件事:

- 轻量 neck 可以与强基线主线稳定接线
- 中断恢复链路可以保持结果连续性

### 5.4 `true Mamba Scheme B` 路线

true Mamba 路线先完成独立环境验证，再进入正式 run。环境验证与稳定性修复过程中记录到的关键信号如下:

- 官方 `mamba-ssm` 与 `causal-conv1d` 已在 `circuitformer-true-mamba` 环境下完成 CUDA 前向验证
- 宿主机环境变量曾出现 `OMP_NUM_THREADS=0`，后续脚本统一显式设为 `1`
- `metrics.py` 中 `np.float` 的兼容性要求推动 `numpy` 固定为 `1.23.5`
- `pytorch-lightning 2.1.0` 的导入路径仍会访问 `pkg_resources`，独立环境据此固定 `setuptools==75.8.0`
- `fp16-mixed` 路线曾出现 `loss_step=nan.0` 与非有限 `BatchNorm` 统计量，后续转入 `bf16-mixed + finite guard + gradient clip`
- 全分辨率 `downsample=1` 路线在显存与早期效果上都偏紧，Scheme B 改用 `downsample=4 + occupancy mask`

最终归档实验目录为 `exp/congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC`，并已补齐 `acceptance_test_epoch99.log`。

## 6. 核心结果

### 6.1 最终分数总览

| 方案 | Val Pearson | Val Spearman | Val Kendall | Test Pearson | Test Spearman | Test Kendall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版首个完整复现 | {formal_val["pearson"]:.4f} | {formal_val["spearman"]:.4f} | {formal_val["kendall"]:.4f} | {formal_test["pearson"]:.4f} | {formal_test["spearman"]:.4f} | {formal_test["kendall"]:.4f} |
| 强基线重跑 | {rerun_val["pearson"]:.4f} | {rerun_val["spearman"]:.4f} | {rerun_val["kendall"]:.4f} | {rerun_test["pearson"]:.4f} | {rerun_test["spearman"]:.4f} | {rerun_test["kendall"]:.4f} |
| zero-init BEV Mamba | {zero_val["pearson"]:.4f} | {zero_val["spearman"]:.4f} | {zero_val["kendall"]:.4f} | {zero_test["pearson"]:.4f} | {zero_test["spearman"]:.4f} | {zero_test["kendall"]:.4f} |
| true Mamba Scheme B | {true_val["pearson"]:.4f} | {true_val["spearman"]:.4f} | {true_val["kendall"]:.4f} | {true_test["pearson"]:.4f} | {true_test["spearman"]:.4f} | {true_test["kendall"]:.4f} |

### 6.2 相对强基线的路线对照

相对强基线，`zero-init BEV Mamba` 的变化如下:

- 验证集: Pearson {zero_gain["val"]["pearson"]["absolute"]:+.4f}，Spearman {zero_gain["val"]["spearman"]["absolute"]:+.4f}，Kendall {zero_gain["val"]["kendall"]["absolute"]:+.4f}
- 测试集: Pearson {zero_gain["test"]["pearson"]["absolute"]:+.4f}，Spearman {zero_gain["test"]["spearman"]["absolute"]:+.4f}，Kendall {zero_gain["test"]["kendall"]["absolute"]:+.4f}

相对强基线，`true Mamba Scheme B` 的变化如下:

- 验证集: Pearson {true_gain["val"]["pearson"]["absolute"]:+.4f}，Spearman {true_gain["val"]["spearman"]["absolute"]:+.4f}，Kendall {true_gain["val"]["kendall"]["absolute"]:+.4f}
- 测试集: Pearson {true_gain["test"]["pearson"]["absolute"]:+.4f}，Spearman {true_gain["test"]["spearman"]["absolute"]:+.4f}，Kendall {true_gain["test"]["kendall"]["absolute"]:+.4f}

从当前单次 run 的指标形态看:

- `zero-init BEV Mamba` 更接近“小幅全指标抬升”
- `true Mamba Scheme B` 更接近“Pearson 接近强基线，排序相关性提升更明显”

### 6.3 参数开销

当前统一参数统计覆盖主线基线与 `zero-init BEV Mamba` 轻量 neck 路线。

- 基线总参数量: `{summary["params"]["baseline"]["total"]:,}`
- `zero-init BEV Mamba` 总参数量: `{summary["params"]["mamba_zero_init"]["total"]:,}`
- 新增参数量: `{param_overhead["absolute"]:,}`，约 `+{param_overhead["relative_percent"]:.3f}%`

这一结果说明，轻量路线观察到的小幅增益与大幅模型扩容无关。

## 7. 图表

### 7.1 验证集 Pearson 曲线

![Validation Pearson](figures/validation_pearson_curves.png)

该图展示了四条实验线的验证终点关系:

- 原版首个复现到强基线之间存在明显跃迁
- `zero-init BEV Mamba` 终点略高于强基线
- `true Mamba Scheme B` 的终点位于强基线附近，并保持另一种指标形态

### 7.2 最终验证/测试指标柱状图

![Final Metric Bars](figures/final_metric_bars.png)

该图给出四条实验线在验证集与测试集上的最终分数。当前 true Mamba 路线的测试结果已经补入同一图中，可直接与前三条线做同口径对照。

### 7.3 相对强基线的净增益

![Gain vs Rerun](figures/gain_vs_rerun.png)

该图将两条 Mamba 路线都放到强基线之上做净增益比较。轻量路线呈现小幅正增益，true Mamba 路线呈现“Pearson 略低、Spearman / Kendall 更高”的对照形态。

### 7.4 数据规模与参数开销

![Dataset And Params](figures/dataset_and_params.png)

该图中的参数对照当前覆盖主线基线与 `zero-init BEV Mamba`。原因较直接: 该图服务于“轻量 neck 的参数增量”这一问题，当前参数统计范围据此限定在轻量路线。

### 7.5 结构示意图

![Architecture Diagram](figures/architecture_diagram.png)

该图对应轻量 `BEV Mamba` 路线。true Mamba 路线使用相同的插入位置，差异集中在 neck 内部的序列混合器实现。

## 8. 事实说明与当前结论

为保证陈述可核验，保留如下事实说明:

- `zero-init` 首轮训练的早期标准输出未完整保留为标准 `train.log`，`epoch 89` 之前的中间验证点当前主要依据已保存 checkpoint 与恢复日志归档。
- 当前公开结论对应 congestion prediction 路线。配置文件中保留了 DRC 入口，当前公开结果未扩展到独立 DRC 实验线。
- 当前训练主线使用带像素权重的 MSE。配置中的 `model.loss` 字段仍保留为代码接口。
- 当前解码器 encoder 部分加载了 `resnet18` 预训练权重，因此当前路线包含外部视觉预训练先验。
- 当前仓库中的两条 Mamba 路线均基于单次 run 归档，相关对照属于单次观察结果。

在这一证据范围内，当前仓库可支撑的结论如下:

- 原版 `CircuitFormer` 复现已经完成
- 强基线训练配方修复带来了本项目中最大的 Pearson 跃迁
- `zero-init BEV Mamba` 路线形成了参数开销很小、指标小幅上扬的轻量 neck 方案
- `true Mamba Scheme B` 路线形成了基于官方 `mamba_ssm.Mamba` 的可运行 neck 方案，并在排序相关性上给出了可对照结果
- 因此，`CircuitFormer-Mamba` 当前可定位为: 高质量复现、工程修复，以及两条 Mamba 相关 neck 路线的并行探索与对照
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
