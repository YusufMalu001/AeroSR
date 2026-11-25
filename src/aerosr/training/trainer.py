"""Reproducible PyTorch training pipeline for aerial super-resolution models."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aerosr.degradation import DegradationConfig, DegradationPipeline
from aerosr.evaluation import compute_psnr, compute_ssim
from aerosr.models import build_model, count_parameters
from aerosr.utils import get_device, get_logger, set_seed

logger = get_logger("aerosr.training")


@dataclass
class TrainConfig:
    """Configuration for training an SR model."""

    model_name: str = "espcn"
    scale_factor: int = 4
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    patch_size: int = 64
    patches_per_image: int = 50
    seed: int = 42
    augment: bool = True
    num_workers: int = 0
    prefer_gpu: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DIORPatchDataset(Dataset):
    """Patch dataset for SR training with deterministic seeding and optional augmentation."""

    def __init__(
        self,
        image_paths: list[Path],
        degradation_cfg: DegradationConfig,
        patches_per_image: int = 50,
        patch_size: int = 64,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.image_paths = [p for p in image_paths if p.exists()]
        self.degradation = DegradationPipeline(degradation_cfg)
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed

        self.images: list[np.ndarray] = []
        for path in self.image_paths:
            img = cv2.imread(str(path))
            if img is not None:
                self.images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_idx = idx // self.patches_per_image
        img = self.images[img_idx]
        h, w = img.shape[:2]

        rng = np.random.default_rng(self.seed + idx)
        y = int(rng.integers(0, max(1, h - self.patch_size)))
        x = int(rng.integers(0, max(1, w - self.patch_size)))
        hr_patch = img[y : y + self.patch_size, x : x + self.patch_size]

        if self.augment:
            if rng.random() < 0.5:
                hr_patch = np.fliplr(hr_patch).copy()
            if rng.random() < 0.5:
                hr_patch = np.flipud(hr_patch).copy()
            k_rot = int(rng.integers(0, 4))
            if k_rot > 0:
                hr_patch = np.rot90(hr_patch, k_rot).copy()

        lr_patch, _ = self.degradation(hr_patch, seed=self.seed + idx)

        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0
        return lr_tensor, hr_tensor


@torch.no_grad()
def validate_sr_model(
    model: nn.Module,
    val_paths: list[Path],
    degradation: DegradationPipeline,
    device: torch.device,
    seed: int = 42,
) -> tuple[float, float]:
    """Evaluate PSNR and SSIM on full validation images."""
    model.eval()
    psnrs: list[float] = []
    ssims: list[float] = []

    for idx, path in enumerate(val_paths):
        hr_bgr = cv2.imread(str(path))
        if hr_bgr is None:
            continue
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        lr_rgb, _ = degradation(hr_rgb, seed=seed + idx)

        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        sr_tensor = model(lr_tensor)
        sr_rgb = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

        psnrs.append(compute_psnr(hr_bgr, sr_bgr))
        ssims.append(compute_ssim(hr_bgr, sr_bgr))

    if not psnrs:
        return 0.0, 0.0
    return float(np.mean(psnrs)), float(np.mean(ssims))


def train_sr_model(
    train_paths: list[Path],
    val_paths: list[Path],
    degradation_cfg: DegradationConfig,
    train_cfg: TrainConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Train an SR model with validation tracking and best-checkpoint saving."""
    set_seed(train_cfg.seed)
    device = get_device(prefer_gpu=train_cfg.prefer_gpu)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DIORPatchDataset(
        image_paths=train_paths,
        degradation_cfg=degradation_cfg,
        patches_per_image=train_cfg.patches_per_image,
        patch_size=train_cfg.patch_size,
        augment=train_cfg.augment,
        seed=train_cfg.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )

    model = build_model(train_cfg.model_name, upscale_factor=train_cfg.scale_factor).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)

    val_degradation = DegradationPipeline(
        DegradationConfig(**{**degradation_cfg.to_dict(), "randomize": False})
    )

    best_psnr = -1.0
    best_ckpt = output_dir / f"{train_cfg.model_name}_best.pt"
    last_ckpt = output_dir / f"{train_cfg.model_name}_last.pt"
    history: list[dict[str, float | int]] = []

    logger.info(
        "Training %s (params=%d) on %d patches for %d epochs on %s",
        train_cfg.model_name,
        count_parameters(model),
        len(dataset),
        train_cfg.epochs,
        device,
    )

    start_time = time.perf_counter()
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for lr, hr in tqdm(dataloader, desc=f"{train_cfg.model_name} Epoch {epoch}/{train_cfg.epochs}", leave=False):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(dataloader))
        val_psnr, val_ssim = validate_sr_model(model, val_paths, val_degradation, device, seed=train_cfg.seed)

        record = {
            "epoch": epoch,
            "train_l1": avg_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        logger.info(
            "Epoch %d/%d | L1: %.4f | Val PSNR: %.2f dB | Val SSIM: %.4f",
            epoch,
            train_cfg.epochs,
            avg_loss,
            val_psnr,
            val_ssim,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), best_ckpt)

    torch.save(model.state_dict(), last_ckpt)
    elapsed = time.perf_counter() - start_time

    summary = {
        "train_config": train_cfg.to_dict(),
        "degradation_config": degradation_cfg.to_dict(),
        "best_val_psnr": best_psnr,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "training_time_sec": elapsed,
        "history": history,
    }
    (output_dir / f"{train_cfg.model_name}_history.json").write_text(json.dumps(summary, indent=2))
    return summary
