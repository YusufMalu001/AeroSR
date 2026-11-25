# AeroSR Architecture

## Overview
AeroSR is a modular PyTorch research framework for **degradation-aware aerial image restoration** and **downstream small-object detection** evaluation.

## Package Structure (`src/aerosr/`)
- `degradation/pipeline.py`: `DegradationConfig` and `DegradationPipeline` supporting configurable downsampling/upsampling, HR/LR Gaussian blur, additive Gaussian noise (with explicit `numpy.random.Generator` seeding), uniform atmospheric haze, and JPEG compression.
- `models/architectures.py` & `models/factory.py`: Implementations of `ESPCN` (inherited), `SRCNN`, `FSRCNN`, and `ResidualSRNet`, plus `build_model`, `count_parameters`, and `model_size_mb`.
- `training/trainer.py`: `DIORPatchDataset` (deterministic cropping + flip/90° rotation augmentation) and `train_sr_model` (L1 loss, Adam, `CosineAnnealingLR`, per-epoch validation PSNR/SSIM, best-checkpoint saving).
- `restoration/restore.py`: `restore_image` and `benchmark_restoration_model` (latency, throughput, parameter count, memory footprint).
- `evaluation/image_metrics.py`: `compute_psnr` and `compute_ssim`.
- `evaluation/detection_metrics.py`: `evaluate_yolo_dataset` (fixed YOLOv8n detector, overall & per-class AP) and `analyze_small_objects` (size-stratified recall, precision, TP/FP/FN).
- `visualization/plots.py`: Publication-ready matplotlib plots for all experiments.
- `utils/`: `set_seed`, `get_device`, `get_logger`.
