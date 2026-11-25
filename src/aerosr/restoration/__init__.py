"""Image restoration inference and efficiency benchmarking."""

from __future__ import annotations

from .restore import benchmark_restoration_model, restore_image

__all__ = ["benchmark_restoration_model", "restore_image"]
