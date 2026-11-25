"""Master experiment runner for AeroSR (Phases 2-8).

Executes:
- Baseline & Experiment B: Train and evaluate ESPCN, SRCNN, FSRCNN, ResidualSRNet vs Clean & Bicubic
- Experiment A: Degradation sensitivity & severity sweep
- Experiment C: Cross-degradation generalization
- Experiment D: Restoration quality (PSNR/SSIM) vs Detection (mAP@0.5)
- Experiment E: Small-object size-stratified analysis
- Experiment F: Computational efficiency benchmarking
- Visualization: Generates all figures in outputs/figures/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from aerosr.degradation import DegradationConfig, DegradationPipeline
from aerosr.evaluation import (
    analyze_small_objects,
    compute_psnr,
    compute_ssim,
    evaluate_yolo_dataset,
)
from aerosr.models import build_model
from aerosr.restoration import benchmark_restoration_model, restore_image
from aerosr.training import TrainConfig, train_sr_model
from aerosr.utils import get_device, get_logger, set_seed
from aerosr.visualization import (
    plot_accuracy_vs_latency,
    plot_degradation_severity,
    plot_metric_bars,
    plot_per_class_ap,
    plot_small_object_recall,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
ORIGINAL_DIR = DATA_DIR / "original"
LABELS_DIR = DATA_DIR / "labels"
WEIGHTS_DIR = WORKSPACE_DIR / "weights"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"

logger = get_logger("aerosr.experiments")


def load_split(split_file: Path) -> list[Path]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    return [ORIGINAL_DIR / line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def prepare_eval_modality(
    image_paths: list[Path],
    out_dir: Path,
    degradation_cfg: DegradationConfig | None = None,
    sr_model: torch.nn.Module | None = None,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> tuple[float, float]:
    """Generate an evaluation dataset directory and return mean (PSNR, SSIM) vs Clean."""
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    pipe = DegradationPipeline(degradation_cfg) if degradation_cfg is not None else None
    psnrs: list[float] = []
    ssims: list[float] = []

    for idx, src_path in enumerate(image_paths):
        hr_bgr = cv2.imread(str(src_path))
        if hr_bgr is None:
            continue

        if pipe is None:
            out_bgr = hr_bgr
        else:
            lr_bgr, upscaled_bgr = pipe(hr_bgr, seed=seed + idx)
            if sr_model is not None:
                out_bgr = restore_image(sr_model, lr_bgr, device=device)
            else:
                out_bgr = upscaled_bgr

        cv2.imwrite(str(img_out / src_path.name), out_bgr)
        lbl_src = LABELS_DIR / f"{src_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy(lbl_src, lbl_out / lbl_src.name)

        if pipe is not None:
            psnrs.append(compute_psnr(hr_bgr, out_bgr))
            ssims.append(compute_ssim(hr_bgr, out_bgr))

    mean_psnr = float(np.mean(psnrs)) if psnrs else float("inf")
    mean_ssim = float(np.mean(ssims)) if ssims else 1.0
    return mean_psnr, mean_ssim


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AeroSR research experiments.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per SR model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--patches-per-image", type=int, default=25, help="Patches per image per epoch.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=True)
    yolo_weights = WEIGHTS_DIR / "yolov8_dior.pt"
    if not yolo_weights.exists():
        raise FileNotFoundError(f"Fixed YOLO weights not found: {yolo_weights}")

    train_paths = load_split(DATA_DIR / "train_images.txt")
    val_paths = load_split(DATA_DIR / "val_images.txt")
    test_paths = load_split(DATA_DIR / "test_images.txt")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_deg = DegradationConfig(
        scale_factor=4,
        apply_blur=True,
        blur_kernel=5,
        blur_in_hr=True,
        apply_noise=True,
        noise_sigma=15.0,
        apply_haze=True,
        haze_t=0.85,
        seed=args.seed,
    )

    results: dict[str, Any] = {
        "seed": args.seed,
        "device": str(device),
        "splits": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
        "experiment_a_degradation": {},
        "experiment_b_restoration": {},
        "experiment_c_generalization": {},
        "experiment_e_small_objects": {},
        "experiment_f_efficiency": {},
    }

    # --- Experiment A: Degradation Sensitivity ---
    logger.info("=== Experiment A: Degradation Sensitivity ===")
    deg_conditions: dict[str, DegradationConfig | None] = {
        "clean": None,
        "downsample_only": DegradationConfig(scale_factor=4, apply_blur=False, apply_noise=False, apply_haze=False, seed=args.seed),
        "blur_only": DegradationConfig(apply_downsample=False, apply_blur=True, blur_kernel=7, apply_noise=False, apply_haze=False, seed=args.seed),
        "noise_only": DegradationConfig(apply_downsample=False, apply_blur=False, apply_noise=True, noise_sigma=25.0, apply_haze=False, seed=args.seed),
        "haze_only": DegradationConfig(apply_downsample=False, apply_blur=False, apply_noise=False, apply_haze=True, haze_t=0.65, seed=args.seed),
        "blur_noise": DegradationConfig(apply_downsample=False, apply_blur=True, blur_kernel=5, apply_noise=True, noise_sigma=15.0, apply_haze=False, seed=args.seed),
        "downsample_blur": DegradationConfig(scale_factor=4, apply_blur=True, blur_kernel=5, apply_noise=False, apply_haze=False, seed=args.seed),
        "downsample_noise": DegradationConfig(scale_factor=4, apply_blur=False, apply_noise=True, noise_sigma=15.0, apply_haze=False, seed=args.seed),
        "combined": base_deg,
    }

    for name, cfg in deg_conditions.items():
        mod_dir = OUTPUTS_DIR / "eval_sets" / f"exp_a_{name}"
        psnr, ssim = prepare_eval_modality(test_paths, mod_dir, degradation_cfg=cfg, seed=args.seed)
        det = evaluate_yolo_dataset(yolo_weights, mod_dir, mod_dir / "dataset.yaml", device=str(device))
        results["experiment_a_degradation"][name] = {"psnr": psnr, "ssim": ssim, **det}
        logger.info("Exp A [%s] -> mAP50: %.4f, mAP50-95: %.4f, PSNR: %.2f", name, det["mAP50"], det["mAP50_95"], psnr)

    # Severity sweep on noise_sigma
    noise_levels = [0.0, 10.0, 20.0, 35.0]
    noise_maps: list[float] = []
    for sigma in noise_levels:
        cfg = DegradationConfig(apply_downsample=False, apply_blur=False, apply_haze=False, apply_noise=sigma > 0, noise_sigma=sigma, seed=args.seed)
        mod_dir = OUTPUTS_DIR / "eval_sets" / f"sweep_noise_{int(sigma)}"
        prepare_eval_modality(test_paths, mod_dir, degradation_cfg=cfg, seed=args.seed)
        det = evaluate_yolo_dataset(yolo_weights, mod_dir, mod_dir / "dataset.yaml", device=str(device))
        noise_maps.append(det["mAP50"])
    results["experiment_a_noise_sweep"] = {"sigma": noise_levels, "mAP50": noise_maps}

    # --- Experiment B & F: Train & Evaluate Restoration Models ---
    logger.info("=== Experiment B & F: Restoration Models ===")
    # Clean & Bicubic baselines
    results["experiment_b_restoration"]["clean"] = results["experiment_a_degradation"]["clean"]
    results["experiment_b_restoration"]["bicubic"] = results["experiment_a_degradation"]["combined"]

    model_names = ["espcn", "srcnn", "fsrcnn", "residual_sr"]
    trained_models: dict[str, torch.nn.Module] = {}

    for m_name in model_names:
        train_cfg = TrainConfig(
            model_name=m_name,
            scale_factor=4,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patches_per_image=args.patches_per_image,
            seed=args.seed,
        )
        ckpt_dir = OUTPUTS_DIR / "checkpoints" / m_name
        train_sr_model(train_paths, val_paths, base_deg, train_cfg, ckpt_dir)

        model = build_model(m_name, upscale_factor=4).to(device)
        model.load_state_dict(torch.load(ckpt_dir / f"{m_name}_best.pt", map_location=device, weights_only=True))
        model.eval()
        trained_models[m_name] = model

        # Evaluate on test set
        mod_dir = OUTPUTS_DIR / "eval_sets" / f"exp_b_{m_name}"
        psnr, ssim = prepare_eval_modality(test_paths, mod_dir, degradation_cfg=base_deg, sr_model=model, device=device, seed=args.seed)
        det = evaluate_yolo_dataset(yolo_weights, mod_dir, mod_dir / "dataset.yaml", device=str(device))
        results["experiment_b_restoration"][m_name] = {"psnr": psnr, "ssim": ssim, **det}

        # Efficiency benchmark
        eff = benchmark_restoration_model(model, input_shape=(1, 3, 200, 200), device=device)
        results["experiment_f_efficiency"][m_name] = eff
        logger.info("Exp B [%s] -> mAP50: %.4f, PSNR: %.2f, SSIM: %.4f, Latency: %.2f ms", m_name, det["mAP50"], psnr, ssim, eff["mean_latency_ms"])

    # --- Experiment C: Cross-Degradation Generalization ---
    logger.info("=== Experiment C: Cross-Degradation Generalization ===")
    unseen_degs = {
        "blur_mismatch_k9": DegradationConfig(scale_factor=4, apply_blur=True, blur_kernel=9, apply_noise=True, noise_sigma=15.0, apply_haze=True, haze_t=0.85, seed=args.seed),
        "noise_mismatch_s30": DegradationConfig(scale_factor=4, apply_blur=True, blur_kernel=5, apply_noise=True, noise_sigma=30.0, apply_haze=True, haze_t=0.85, seed=args.seed),
        "bilinear_downsample": DegradationConfig(scale_factor=4, downsample_mode="bilinear", apply_blur=True, blur_kernel=5, apply_noise=True, noise_sigma=15.0, apply_haze=True, haze_t=0.85, seed=args.seed),
        "heavy_haze_t06": DegradationConfig(scale_factor=4, apply_blur=True, blur_kernel=5, apply_noise=True, noise_sigma=15.0, apply_haze=True, haze_t=0.60, seed=args.seed),
        "jpeg_q50": DegradationConfig(scale_factor=4, apply_blur=True, blur_kernel=5, apply_noise=True, noise_sigma=15.0, apply_haze=True, haze_t=0.85, apply_jpeg=True, jpeg_quality=50, seed=args.seed),
    }

    for deg_name, deg_cfg in unseen_degs.items():
        results["experiment_c_generalization"][deg_name] = {}
        # Bicubic
        b_dir = OUTPUTS_DIR / "eval_sets" / f"exp_c_{deg_name}_bicubic"
        psnr, ssim = prepare_eval_modality(test_paths, b_dir, degradation_cfg=deg_cfg, seed=args.seed)
        det = evaluate_yolo_dataset(yolo_weights, b_dir, b_dir / "dataset.yaml", device=str(device))
        results["experiment_c_generalization"][deg_name]["bicubic"] = {"psnr": psnr, "ssim": ssim, **det}

        for m_name, model in trained_models.items():
            m_dir = OUTPUTS_DIR / "eval_sets" / f"exp_c_{deg_name}_{m_name}"
            psnr, ssim = prepare_eval_modality(test_paths, m_dir, degradation_cfg=deg_cfg, sr_model=model, device=device, seed=args.seed)
            det = evaluate_yolo_dataset(yolo_weights, m_dir, m_dir / "dataset.yaml", device=str(device))
            results["experiment_c_generalization"][deg_name][m_name] = {"psnr": psnr, "ssim": ssim, **det}

    # --- Experiment E: Small-Object Performance ---
    logger.info("=== Experiment E: Small-Object Analysis ===")
    for modality in ["clean", "bicubic", *model_names]:
        if modality == "clean":
            mod_dir = OUTPUTS_DIR / "eval_sets" / "exp_a_clean"
        elif modality == "bicubic":
            mod_dir = OUTPUTS_DIR / "eval_sets" / "exp_a_combined"
        else:
            mod_dir = OUTPUTS_DIR / "eval_sets" / f"exp_b_{modality}"
        results["experiment_e_small_objects"][modality] = analyze_small_objects(
            yolo_weights, mod_dir / "images", mod_dir / "labels", device=str(device)
        )

    # Save all results
    results_file = OUTPUTS_DIR / "all_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    logger.info("Saved all results to %s", results_file)

    # --- Generate Plots ---
    deg_labels = list(results["experiment_a_degradation"].keys())
    deg_maps = [results["experiment_a_degradation"][k]["mAP50"] for k in deg_labels]
    plot_metric_bars(deg_labels, deg_maps, "Experiment A: Detection mAP@0.5 by Degradation", "mAP@0.5", figures_dir / "exp_a_degradation_map50.png")

    plot_degradation_severity(
        {"Gaussian Noise": (noise_levels, noise_maps)},
        xlabel="Noise Sigma",
        ylabel="mAP@0.5",
        title="Detection mAP@0.5 vs. Noise Severity",
        output_path=figures_dir / "exp_a_noise_severity.png",
    )

    rest_labels = ["clean", "bicubic", *model_names]
    rest_maps = [results["experiment_b_restoration"][k]["mAP50"] for k in rest_labels]
    plot_metric_bars(rest_labels, rest_maps, "Experiment B: Detection mAP@0.5 by Restoration Method", "mAP@0.5", figures_dir / "exp_b_restoration_map50.png")

    sr_only = ["bicubic", *model_names]
    plot_metric_bars(sr_only, [results["experiment_b_restoration"][k]["psnr"] for k in sr_only], "PSNR Comparison (dB)", "PSNR (dB)", figures_dir / "exp_b_psnr.png")
    plot_metric_bars(sr_only, [results["experiment_b_restoration"][k]["ssim"] for k in sr_only], "SSIM Comparison", "SSIM", figures_dir / "exp_b_ssim.png")

    plot_accuracy_vs_latency(
        model_names,
        [results["experiment_f_efficiency"][m]["mean_latency_ms"] for m in model_names],
        [results["experiment_b_restoration"][m]["mAP50"] for m in model_names],
        figures_dir / "exp_f_accuracy_vs_latency.png",
    )

    plot_small_object_recall(
        rest_labels,
        [results["experiment_e_small_objects"][k]["small"]["recall"] for k in rest_labels],
        [results["experiment_e_small_objects"][k]["medium"]["recall"] for k in rest_labels],
        [results["experiment_e_small_objects"][k]["large"]["recall"] for k in rest_labels],
        figures_dir / "exp_e_small_object_recall.png",
    )

    plot_per_class_ap(
        {k: results["experiment_b_restoration"][k].get("per_class_ap50", {}) for k in ["clean", "bicubic", "espcn", "residual_sr"]},
        figures_dir / "exp_b_per_class_ap.png",
    )
    logger.info("All figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()
