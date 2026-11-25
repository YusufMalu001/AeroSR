"""Research visualization and plotting utilities."""

from .plots import (
    plot_accuracy_vs_latency,
    plot_degradation_severity,
    plot_metric_bars,
    plot_per_class_ap,
    plot_small_object_recall,
)

__all__ = [
    "plot_accuracy_vs_latency",
    "plot_degradation_severity",
    "plot_metric_bars",
    "plot_per_class_ap",
    "plot_small_object_recall",
]
