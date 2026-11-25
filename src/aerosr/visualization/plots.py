"""Publication-quality matplotlib plots for AeroSR experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_bars(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: Path | str,
) -> Path:
    """Save a bar chart comparing a single metric across methods or conditions."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="#2b6cb0", edgecolor="#1a365d", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_degradation_severity(
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path | str,
) -> Path:
    """Plot detection or image quality metric vs. degradation severity."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (xs, ys) in curves.items():
        ax.plot(xs, ys, marker="o", linewidth=2, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_accuracy_vs_latency(
    methods: Sequence[str],
    latencies_ms: Sequence[float],
    maps: Sequence[float],
    output_path: Path | str,
    title: str = "Detection mAP@0.5 vs. Restoration Latency",
) -> Path:
    """Scatter plot of mAP@0.5 against inference latency."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(latencies_ms, maps, s=90, color="#c53030", zorder=3)
    for m, x, y in zip(methods, latencies_ms, maps):
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Restoration Latency (ms / image)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_small_object_recall(
    methods: Sequence[str],
    small_recalls: Sequence[float],
    medium_recalls: Sequence[float],
    large_recalls: Sequence[float],
    output_path: Path | str,
) -> Path:
    """Grouped bar chart of recall by object size category."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, small_recalls, width, label="Small (<0.5% area)")
    ax.bar(x, medium_recalls, width, label="Medium (0.5%-5% area)")
    ax.bar(x + width, large_recalls, width, label="Large (>=5% area)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title("Object Detection Recall by Object Size")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_class_ap(
    class_ap_by_method: Mapping[str, Mapping[str, float]],
    output_path: Path | str,
) -> Path:
    """Plot per-class AP@0.5 across methods."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    methods = list(class_ap_by_method.keys())
    classes = sorted({c for m in methods for c in class_ap_by_method[m].keys()})
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(methods):
        vals = [class_ap_by_method[m].get(c, 0.0) for c in classes]
        ax.bar(x + (i - len(methods) / 2.0 + 0.5) * width, vals, width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP@0.5 Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
