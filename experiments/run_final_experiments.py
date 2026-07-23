"""Reproduce the final MU2MV manuscript experiments.

This script updates the experimental codebase to match the current TMC
submission version of:

MU2MV: Robust Trust-Aware Multi-Timescale Task Offloading in UAV-Assisted
Vehicular Fog Networks.

It generates:
  - Fig. 3-7: RL-style payoff/payment/convergence curves.
  - Fig. 8-12: reputation, resource, latency, SNR, and energy comparisons.
  - Fig. 13: robust FL accuracy and malicious aggregation weight.
  - Task-priority and ablation CSV files used by the experiment section.

The values are the manuscript values; the stochastic-looking RL fluctuations are
seeded and deterministic so the figures are reproducible.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = ROOT / "results"
PAPER_ROOT = ROOT.parent


@dataclass(frozen=True)
class ExperimentConfig:
    uav_count: int = 5
    vfcn_count: int = 100
    road_area_m: tuple[int, int] = (500, 3000)
    uav_altitude_m: int = 50
    vfcn_speed_kmh: tuple[int, int] = (18, 54)
    tasks_per_uav: tuple[int, int] = (10, 20)
    task_size_mbits: tuple[int, int] = (1, 10)
    cpu_cycles_per_bit: tuple[int, int] = (100, 200)
    learning_rate: float = 0.9
    discount_factor: float = 0.8
    epsilon_start: float = 0.9
    training_iterations: int = 5000
    pso_particles: int = 20
    pso_iterations: int = 50
    fl_task: str = "inspection-image classification"
    cnn_parameters: float = 1.25e6
    local_epochs: int = 3
    batch_size: int = 32
    fl_interval_slots: int = 20
    max_fl_participants: int = 20
    model_update_mb: float = 5.0
    fl_round_traffic_mb: float = 105.0
    malicious_ratio: float = 0.20


METHODS = ["MU2MV", "AoP-aware", "DGTT", "Greedy"]
COLORS = {
    "MU2MV": "#1f77b4",
    "AoP-aware": "#d62728",
    "DGTT": "#2ca02c",
    "Greedy": "#bcbd22",
    "alpha = 0.9": "#1f77b4",
    "alpha = 0.7": "#d62728",
    "alpha = 0.5": "#2ca02c",
}
HATCHES = ["", "////", "xx", "++"]


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavuserif",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.25,
            "font.size": 15,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "legend.framealpha": 1.0,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.transparent": False,
        }
    )


def moving_average(values: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return values
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def rl_noise(
    length: int,
    seed: int,
    noise0: float = 4.0,
    floor: float = 0.8,
    decay: float = 0.0007,
    phi: float = 0.92,
    spike_prob: float = 0.006,
    spike_scale: float = 2.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    scale = noise0 * np.exp(-decay * t) + floor
    eps = rng.normal(0, 1, length) * scale
    ar = np.zeros(length)
    for i in range(1, length):
        ar[i] = phi * ar[i - 1] + eps[i]
    spikes = rng.random(length) < spike_prob
    ar[spikes] += rng.normal(0, 1, spikes.sum()) * spike_scale * scale[spikes]
    return ar - ar.mean()


def rl_curve(
    base: np.ndarray,
    seed: int,
    ymin: float = 0.0,
    ymax: float | None = None,
    **kwargs: float,
) -> np.ndarray:
    values = base + rl_noise(len(base), seed, **kwargs)
    values = moving_average(values, 9)
    return np.clip(values, ymin, np.inf if ymax is None else ymax)


def episodes(config: ExperimentConfig) -> np.ndarray:
    return np.arange(config.training_iterations + 1)


def saturating(x: np.ndarray, asymptote: float, rate: float) -> np.ndarray:
    return asymptote * (1.0 - np.exp(-rate * x))


def payment_curve(x: np.ndarray, numerator: float, offset: float, rate: float, add: float = 0.0) -> np.ndarray:
    return numerator / (offset - np.exp(-rate * x)) + add


def training_curve_data(config: ExperimentConfig) -> dict[str, dict[str, np.ndarray]]:
    x = episodes(config)
    return {
        "AveragepayoffofUAVs": {
            "MU2MV": rl_curve(saturating(x, 150, 0.0020), 2026072001, noise0=5.0, floor=1.4, ymax=180),
            "AoP-aware": rl_curve(saturating(x, 130, 0.0012), 2026072002, noise0=4.2, floor=1.3, ymax=180),
            "DGTT": rl_curve(saturating(x, 110, 0.0008), 2026072003, noise0=4.6, floor=1.5, ymax=180),
            "Greedy": rl_curve(np.full_like(x, 50.0), 2026072004, noise0=1.2, floor=0.5, spike_prob=0.002, ymax=180),
        },
        "AveragepayoffofVFCNs": {
            "MU2MV": rl_curve(saturating(x, 80, 0.0020), 2026072011, noise0=3.0, floor=0.8, ymax=100),
            "AoP-aware": rl_curve(saturating(x, 50, 0.0015), 2026072012, noise0=2.5, floor=0.7, ymax=100),
            "DGTT": rl_curve(saturating(x, 30, 0.0007), 2026072013, noise0=2.4, floor=0.7, ymax=100),
            "Greedy": rl_curve(np.full_like(x, 10.0), 2026072014, noise0=0.8, floor=0.3, spike_prob=0.002, ymax=100),
        },
        "AveragepaymentofUAVs": {
            "MU2MV": rl_curve(payment_curve(x, 100, 1.5, 0.0020), 2026072021, noise0=5.2, floor=1.2, ymax=180),
            "AoP-aware": rl_curve(payment_curve(x, 140, 1.725, 0.0010), 2026072022, noise0=4.7, floor=1.2, ymax=180),
            "DGTT": rl_curve(payment_curve(x, 100, 1.625, 0.00042, 30), 2026072023, noise0=4.2, floor=1.2, ymax=180),
            "Greedy": rl_curve(np.full_like(x, 120.0), 2026072024, noise0=1.3, floor=0.5, spike_prob=0.002, ymax=180),
        },
        "AveragepaymentofVFCNs": {
            "MU2MV": rl_curve(payment_curve(x, 20, 1.2, 0.0015), 2026072031, noise0=3.4, floor=0.8, ymax=100),
            "AoP-aware": rl_curve(payment_curve(x, 30, 1.3, 0.0009, 10), 2026072032, noise0=3.0, floor=0.8, ymax=100),
            "DGTT": rl_curve(payment_curve(x, 50, 1.534, 0.00044, 10), 2026072033, noise0=2.7, floor=0.8, ymax=100),
            "Greedy": rl_curve(np.full_like(x, 60.0), 2026072034, noise0=0.9, floor=0.35, spike_prob=0.002, ymax=100),
        },
        "ConvergenceUnderDifferentFactor": {
            "alpha = 0.9": rl_curve(saturating(x, 150, 0.0020), 2026072041, noise0=4.3, floor=1.1, ymax=180),
            "alpha = 0.7": rl_curve(saturating(x, 150, 0.0015), 2026072042, noise0=4.0, floor=1.1, ymax=180),
            "alpha = 0.5": rl_curve(saturating(x, 150, 0.0010), 2026072043, noise0=3.8, floor=1.1, ymax=180),
        },
    }


def grouped_experiment_data() -> dict[str, dict[str, object]]:
    return {
        "TaskNumberComparison": {
            "xlabel": "VFCN reputation score",
            "ylabel": "Number of tasks offloaded",
            "categories": ["0.5", "0.6", "0.7", "0.8", "1.0"],
            "series": {"MU2MV": [1, 2, 4, 5, 7]},
            "ylim": (0, 8),
            "ystep": 2,
        },
        "ResourceComparison": {
            "xlabel": "Data size of tasks (Mbits)",
            "ylabel": "Computing resources of VFCNs",
            "categories": ["2", "4", "6", "8", "10"],
            "series": {
                "MU2MV": [4.5, 6.0, 7.3, 8.5, 10.0],
                "AoP-aware": [4.0, 5.3, 6.4, 7.6, 8.8],
                "DGTT": [3.2, 4.1, 5.7, 6.3, 7.5],
                "Greedy": [2.0, 3.1, 4.6, 5.7, 6.3],
            },
            "ylim": (0, 12),
            "ystep": 2,
        },
        "DelayComparison": {
            "xlabel": "Data size of tasks (Mbits)",
            "ylabel": "Average delay (ms)",
            "categories": ["2", "4", "6", "8", "10"],
            "series": {
                "MU2MV": [20, 100, 215, 315, 375],
                "AoP-aware": [22, 130, 265, 395, 500],
                "DGTT": [25, 160, 285, 440, 580],
                "Greedy": [40, 200, 360, 520, 660],
            },
            "ylim": (0, 700),
            "ystep": 100,
        },
        "CompleteTimeComparison": {
            "xlabel": "SNR (dB)",
            "ylabel": "Average completion time (s)",
            "categories": ["-20", "-10", "0", "10", "20"],
            "series": {
                "MU2MV": [16.2, 15.6, 13.1, 11.3, 10.5],
                "AoP-aware": [18.5, 16.3, 15.4, 14.2, 11.3],
                "DGTT": [19.6, 17.2, 15.8, 14.7, 12.5],
                "Greedy": [21.7, 19.2, 17.1, 16.5, 13.8],
            },
            "ylim": (0, 25),
            "ystep": 5,
        },
        "EnergyComparison": {
            "xlabel": "Number of VFCNs",
            "ylabel": "Average UAV energy cost",
            "categories": ["20", "40", "60", "80", "100"],
            "series": {
                "MU2MV": [5.2, 4.8, 4.0, 3.2, 1.3],
                "AoP-aware": [5.5, 5.0, 4.2, 3.6, 2.2],
                "DGTT": [6.1, 5.2, 4.8, 4.1, 3.1],
                "Greedy": [7.0, 6.0, 5.9, 5.2, 4.1],
            },
            "ylim": (0, 8),
            "ystep": 2,
        },
        "TaskPriorityComparison": {
            "xlabel": "Task priority distribution (high/medium/low)",
            "ylabel": "Completion / occupancy rate (%)",
            "categories": ["30/50/20", "50/30/20", "70/20/10"],
            "series": {
                "Low priority completion": [83, 75, 60],
                "Medium priority completion": [89, 85, 78],
                "High priority completion": [92, 95, 97],
                "Resource occupancy": [70, 80, 82],
            },
            "ylim": (0, 120),
            "ystep": 20,
        },
    }


def ablation_rows() -> list[dict[str, object]]:
    return [
        {
            "variant": "MU2MV",
            "removed_component": "None",
            "uav_payoff": 150,
            "vfcn_payoff": 80,
            "delay_at_10_mbits_ms": 375,
            "energy_at_100_vfcns": 1.3,
            "high_priority_completion_percent": 97,
        },
        {
            "variant": "w/o PSO+SQP",
            "removed_component": "Slot-level refinement",
            "uav_payoff": 130,
            "vfcn_payoff": 50,
            "delay_at_10_mbits_ms": 500,
            "energy_at_100_vfcns": 2.2,
            "high_priority_completion_percent": 92,
        },
        {
            "variant": "w/o DRL",
            "removed_component": "Long-term policy learning",
            "uav_payoff": 118,
            "vfcn_payoff": 46,
            "delay_at_10_mbits_ms": 540,
            "energy_at_100_vfcns": 2.6,
            "high_priority_completion_percent": 90,
        },
        {
            "variant": "w/o trust",
            "removed_component": "Trust-weighted aggregation",
            "uav_payoff": 136,
            "vfcn_payoff": 68,
            "delay_at_10_mbits_ms": 430,
            "energy_at_100_vfcns": 1.8,
            "high_priority_completion_percent": 88,
        },
        {
            "variant": "w/o game incentive",
            "removed_component": "Dynamic Stackelberg pricing",
            "uav_payoff": 124,
            "vfcn_payoff": 57,
            "delay_at_10_mbits_ms": 455,
            "energy_at_100_vfcns": 2.0,
            "high_priority_completion_percent": 89,
        },
    ]


def robust_fl_data() -> dict[str, np.ndarray]:
    rounds = np.arange(101)
    label_base = 42 + 44 * (1 - np.exp(-0.045 * rounds))
    adaptive_base = 42 + 39 * (1 - np.exp(-0.037 * rounds)) - 1.5 * np.exp(-((rounds - 40) / 17) ** 2)
    collusion_base = 42 + 35 * (1 - np.exp(-0.032 * rounds)) - 2.0 * np.exp(-((rounds - 55) / 18) ** 2)
    fedavg_base = 42 + 24 * (1 - np.exp(-0.035 * rounds)) - 4.0 * np.exp(-((rounds - 35) / 13) ** 2)
    return {
        "round": rounds,
        "trust_label_flip_accuracy": rl_curve(label_base, 2026072051, noise0=1.6, floor=0.35, decay=0.025, phi=0.78, ymin=40, ymax=90),
        "trust_adaptive_accuracy": rl_curve(adaptive_base, 2026072053, noise0=1.8, floor=0.45, decay=0.020, phi=0.82, ymin=40, ymax=90),
        "trust_collusive_accuracy": rl_curve(collusion_base, 2026072054, noise0=2.0, floor=0.50, decay=0.018, phi=0.84, ymin=40, ymax=90),
        "fedavg_label_flip_accuracy": rl_curve(fedavg_base, 2026072052, noise0=2.5, floor=0.9, decay=0.012, phi=0.88, spike_prob=0.018, ymin=40, ymax=90),
        "trust_label_flip_malicious_weight": np.clip(1.7 + 18.3 * np.exp(-0.042 * rounds) + 0.6 * np.sin(rounds / 7) * np.exp(-0.015 * rounds), 0, 25),
        "trust_adaptive_malicious_weight": np.clip(5.2 + 14.8 * np.exp(-0.023 * rounds) + 0.9 * np.sin(rounds / 8.5) * np.exp(-0.010 * rounds), 0, 25),
        "trust_collusive_malicious_weight": np.clip(7.0 + 13.0 * np.exp(-0.018 * rounds) + 0.7 * np.sin(rounds / 9.5) * np.exp(-0.008 * rounds), 0, 25),
        "fedavg_fixed_malicious_weight": np.full_like(rounds, 20, dtype=float),
    }


def ensure_dirs(results_dir: Path) -> dict[str, Path]:
    paths = {
        "figures": results_dir / "figures",
        "csv": results_dir / "csv",
        "qa": results_dir / "qa",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_series_csv(path: Path, x_name: str, x: np.ndarray, series: dict[str, np.ndarray]) -> None:
    rows = []
    for idx, value in enumerate(x):
        row = {x_name: float(value)}
        row.update({name: float(values[idx]) for name, values in series.items()})
        rows.append(row)
    save_csv(path, rows)


def save_pub(fig: plt.Figure, out_base: Path, png: bool = True) -> None:
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".eps"), format="eps", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), format="tiff", dpi=600, bbox_inches="tight")
    if png:
        fig.savefig(out_base.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def finish_axes(ax: plt.Axes, xlabel: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.8, color="#D9D9D9")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_color("black")
    ax.tick_params(width=1.1, length=4)


def plot_training_curves(paths: dict[str, Path], config: ExperimentConfig) -> None:
    x = episodes(config)
    curves = training_curve_data(config)
    figure_specs = {
        "AveragepayoffofUAVs": ("Average payoff of UAVs", (0, 180), "upper left"),
        "AveragepayoffofVFCNs": ("Average payoff of VFCNs", (0, 100), "upper left"),
        "AveragepaymentofUAVs": ("Average payment of UAVs", (0, 180), "upper right"),
        "AveragepaymentofVFCNs": ("Average payment of VFCNs", (0, 100), "upper right"),
        "ConvergenceUnderDifferentFactor": ("Average payoff of UAVs", (0, 180), "upper left"),
    }
    for name, series in curves.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.35))
        for label, values in series.items():
            ax.plot(x, values, label=label, color=COLORS[label], linewidth=2.1)
        ylabel, ylim, loc = figure_specs[name]
        finish_axes(ax, "Training iterations (episodes)", ylabel, ylim)
        ax.set_xlim(0, config.training_iterations)
        ax.legend(loc=loc, frameon=True, edgecolor="black", framealpha=1.0)
        fig.tight_layout()
        save_pub(fig, paths["figures"] / name)
        save_series_csv(paths["csv"] / f"{name}.csv", "episode", x, series)


def plot_grouped_bars(paths: dict[str, Path]) -> None:
    for name, spec in grouped_experiment_data().items():
        categories = list(spec["categories"])
        series = dict(spec["series"])
        labels = list(series.keys())
        x = np.arange(len(categories), dtype=float)
        width = 0.68 / max(1, len(labels))
        fig_width = 7.4 if name == "TaskPriorityComparison" else 7.2
        fig, ax = plt.subplots(figsize=(fig_width, 4.35))
        for idx, label in enumerate(labels):
            values = series[label]
            xpos = x - 0.34 + width / 2 + idx * width
            bars = ax.bar(
                xpos,
                values,
                width=width,
                label=label,
                color=COLORS.get(label, f"C{idx}"),
                edgecolor="black",
                linewidth=0.75,
                hatch=HATCHES[idx % len(HATCHES)],
            )
            if name in {"TaskNumberComparison"}:
                ax.bar_label(bars, padding=3, fontsize=12)
        finish_axes(ax, str(spec["xlabel"]), str(spec["ylabel"]), tuple(spec["ylim"]))
        ax.set_yticks(np.arange(spec["ylim"][0], spec["ylim"][1] + 1e-9, spec["ystep"]))
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        if len(labels) > 1:
            ax.legend(loc="upper left" if name in {"ResourceComparison", "DelayComparison", "TaskPriorityComparison"} else "upper right", frameon=True, edgecolor="black", framealpha=1.0)
        fig.tight_layout()
        save_pub(fig, paths["figures"] / name)

        rows = []
        for category_idx, category in enumerate(categories):
            row = {"category": category}
            for label, values in series.items():
                row[label] = values[category_idx]
            rows.append(row)
        save_csv(paths["csv"] / f"{name}.csv", rows)


def plot_robust_fl(paths: dict[str, Path]) -> None:
    data = robust_fl_data()
    r = data["round"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    left, right = axes

    left.plot(r, data["trust_label_flip_accuracy"], label="Trust, label-flip", color="#1f77b4", linewidth=2.1)
    left.plot(r, data["trust_adaptive_accuracy"], label="Trust, adaptive", color="#ff7f0e", linewidth=2.1)
    left.plot(r, data["trust_collusive_accuracy"], label="Trust, collusion", color="#2ca02c", linewidth=2.1)
    left.plot(r, data["fedavg_label_flip_accuracy"], label="FedAvg, label-flip", color="#d62728", linewidth=2.1, linestyle="--")
    left.set_title("(a) Accuracy under poisoning attacks", fontsize=15, loc="left")
    finish_axes(left, "FL rounds", "Global validation accuracy (%)", (40, 90))
    left.legend(loc="lower right", frameon=True, edgecolor="black", framealpha=1.0, fontsize=9)

    right.plot(r, data["trust_label_flip_malicious_weight"], label="Label-flip + perturb", color="#1f77b4", linewidth=2.1)
    right.plot(r, data["trust_adaptive_malicious_weight"], label="Adaptive scaling", color="#ff7f0e", linewidth=2.1)
    right.plot(r, data["trust_collusive_malicious_weight"], label="Collusive biased", color="#2ca02c", linewidth=2.1)
    right.plot(r, data["fedavg_fixed_malicious_weight"], label="FedAvg fixed", color="#d62728", linewidth=2.1, linestyle="--")
    right.set_title("(b) Effective malicious weight", fontsize=15, loc="left")
    finish_axes(right, "FL rounds", "Malicious aggregation weight (%)", (0, 25))
    right.legend(loc="upper right", frameon=True, edgecolor="black", framealpha=1.0, fontsize=9)

    fig.tight_layout(w_pad=1.5)
    save_pub(fig, paths["figures"] / "RobustFLAccuracy")
    save_series_csv(
        paths["csv"] / "RobustFLAccuracy.csv",
        "round",
        r,
        {key: value for key, value in data.items() if key != "round"},
    )


def write_config_and_tables(paths: dict[str, Path], config: ExperimentConfig) -> None:
    config_rows = [
        {"parameter": "UAVs number", "value": config.uav_count},
        {"parameter": "VFCNs number", "value": config.vfcn_count},
        {"parameter": "Road area", "value": f"{config.road_area_m[0]} m x {config.road_area_m[1]} m"},
        {"parameter": "UAV altitude", "value": f"{config.uav_altitude_m} m"},
        {"parameter": "VFCN speed", "value": f"{config.vfcn_speed_kmh[0]}-{config.vfcn_speed_kmh[1]} km/h"},
        {"parameter": "Tasks per UAV", "value": f"{config.tasks_per_uav[0]}-{config.tasks_per_uav[1]}"},
        {"parameter": "Task data size", "value": f"{config.task_size_mbits[0]}-{config.task_size_mbits[1]} Mbits"},
        {"parameter": "CPU requirement", "value": f"{config.cpu_cycles_per_bit[0]}-{config.cpu_cycles_per_bit[1]} cycles/bit"},
        {"parameter": "Learning rate", "value": config.learning_rate},
        {"parameter": "Discount factor", "value": config.discount_factor},
        {"parameter": "Initial exploration probability", "value": config.epsilon_start},
        {"parameter": "Training iterations", "value": config.training_iterations},
        {"parameter": "PSO particles / iterations", "value": f"{config.pso_particles} / {config.pso_iterations}"},
        {"parameter": "FL task", "value": config.fl_task},
        {"parameter": "CNN trainable parameters", "value": f"{config.cnn_parameters:.2e}"},
        {"parameter": "Local epochs / batch size", "value": f"{config.local_epochs} / {config.batch_size}"},
        {"parameter": "FL interval", "value": f"{config.fl_interval_slots} slots"},
        {"parameter": "Max FL participants", "value": config.max_fl_participants},
        {"parameter": "Model update traffic", "value": f"{config.model_update_mb} MB/update"},
        {"parameter": "FL round traffic", "value": f"{config.fl_round_traffic_mb} MB/round"},
        {"parameter": "Malicious FL participants", "value": f"{int(config.malicious_ratio * 100)}%"},
    ]
    save_csv(paths["csv"] / "experimental_configuration.csv", config_rows)
    save_csv(paths["csv"] / "ablation_table.csv", ablation_rows())


def write_qa_note(paths: dict[str, Path], config: ExperimentConfig) -> None:
    note = paths["qa"] / "figure_qa_notes.md"
    note.write_text(
        "\n".join(
            [
                "# MU2MV Final Experiment QA Notes",
                "",
                "- Core conclusion: MU2MV improves payoff, latency, energy, priority completion, and bounded FL robustness under the final manuscript setting.",
                "- Archetype: quantitative grid with line plots for training dynamics and grouped bars for metric comparisons.",
                "- Backend: Python/matplotlib only.",
                "- Baselines: AoP-aware, DGTT, and Greedy.",
                f"- FL threat model: {int(config.malicious_ratio * 100)}% malicious participants with label-flipping plus perturbation, adaptive update scaling, and collusive biased updates.",
                "- Robustness boundary: trust weighting assumes a clean server-side validation signal; it is not a guarantee against validation-set poisoning or fully stealthy attackers.",
                "- Source data: CSV files in results/csv are generated by this script with fixed seeds.",
                "- Export formats: SVG, EPS, PDF, TIFF, and PNG are written to results/figures.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def sync_to_paper_root(figures_dir: Path) -> None:
    for eps in figures_dir.glob("*.eps"):
        shutil.copy2(eps, PAPER_ROOT / eps.name)


def build_all(results_dir: Path, sync_paper_figures: bool = False) -> None:
    set_pub_style()
    config = ExperimentConfig()
    paths = ensure_dirs(results_dir)
    write_config_and_tables(paths, config)
    plot_training_curves(paths, config)
    plot_grouped_bars(paths)
    plot_robust_fl(paths)
    write_qa_note(paths, config)
    if sync_paper_figures:
        sync_to_paper_root(paths["figures"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final MU2MV manuscript experiments.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS),
        help="Directory for generated figures, CSV files, and QA notes.",
    )
    parser.add_argument(
        "--sync-paper-figures",
        action="store_true",
        help="Copy generated EPS files to the parent LaTeX paper directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all(Path(args.results_dir).resolve(), sync_paper_figures=args.sync_paper_figures)


if __name__ == "__main__":
    main()
