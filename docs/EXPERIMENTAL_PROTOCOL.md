# Experimental Protocol

## 1. Dataset & Leakage-Free Splits
- **Source:** DIOR aerial object detection dataset (`JPEGImages-test.zip` + `Annotations.zip`), 300-image subset sampled with `seed=42`.
- **Splits (`scripts/prepare_splits.py`):** Image-level split (no shared source images across splits):
  - **Train:** 160 images (`data/train_images.txt`) — used exclusively for SR patch extraction and training.
  - **Validation:** 40 images (`data/val_images.txt`) — used for per-epoch PSNR/SSIM model selection.
  - **Test:** 100 images (`data/test_images.txt`) — held out for all final restoration, detection, and generalization evaluations.

## 2. Fixed Downstream Detector
- **Model:** `weights/yolov8_dior.pt` (YOLOv8n trained on DIOR, 20 classes, `imgsz=800`).
- **Invariance:** Detector weights, confidence threshold (`0.25` for box-level analysis), IoU threshold (`0.5`), and image IDs are held strictly constant across all modalities.

## 3. Small-Object Categorization
Objects are categorized by bounding-box area relative to image area (`box_area / image_area`):
- **Small:** `< 0.005` (< 0.5% of image area)
- **Medium:** `[0.005, 0.05)`
- **Large:** `>= 0.05`

## 4. Reproducibility
All experiments use `seed=42` across Python, NumPy (`default_rng`), and PyTorch. Configurations and results are persisted to `outputs/all_results.json`.
