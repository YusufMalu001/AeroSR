# AeroSR Project Audit

**Audit Date:** 2026-11-25  
**Repository:** `/home/yusuf/Desktop/AeroSR`  
**Scope:** Initial read-only audit of all existing source files, weights, configuration, and git history prior to any code modifications.

---

## 1. Repository Overview & Git History

The repository consists of 7 Python scripts, 1 weights file (`weights/yolov8_dior.pt`, ~6.29 MB), a `frontend/` directory (`index.html`, `app.js`, `style.css`), `README.md`, and `.gitignore`. There is no `requirements.txt`, `pyproject.toml`, `tests/` directory, `data/` directory, or `weights/espcn_dior.pt` checkpoint currently present in the workspace.

Git history shows single-file initial commits (`updated app.js`, `updated index.html`, `updated style.css`, `updated prepare_dataset.py`, `updated server.py`, `updated sr_model.py`, `updated test_download.py`, `updated train_sr.py`, `updated yolov8_dior.pt`, `updated README.md`).

---

## 2. Detailed Audit Findings

### 2.1 Original Architecture
The pipeline consists of four decoupled stages:
1. **Dataset Preparation (`prepare_dataset.py`)**: Downloads DIOR test images (`JPEGImages-test.zip`) and XML annotations (`Annotations.zip`) from Google Drive, extracts a 300-image subset, converts Pascal VOC XML bounding boxes to YOLO format, and splits into 200 train / 100 eval images.
2. **Synthetic Degradation (`degrade.py`)**: Applies 4x bicubic downsampling, uniform atmospheric haze, Gaussian blur, and additive Gaussian noise, plus bicubic upsampling for a baseline.
3. **Super-Resolution Training (`sr_model.py`, `train_sr.py`)**: Trains an ESPCN model on random 64×64 HR patches (degraded on-the-fly to 16×16 LR patches).
4. **Evaluation & Visualization (`evaluate.py`, `server.py`, `frontend/`)**: Evaluates YOLOv8n on `original`, `degraded` (bicubic-upscaled), and `sr_enhanced` image sets, and serves an interactive FastAPI + HTML/JS dashboard.

### 2.2 ESPCN Implementation Details (`sr_model.py`)
- **Class:** `ESPCN(upscale_factor=4, channels=3)`
- **Layers:**
  - `conv1`: `Conv2d(3, 64, kernel_size=5, padding=2)` + `torch.tanh`
  - `conv2`: `Conv2d(64, 32, kernel_size=3, padding=1)` + `torch.tanh`
  - `conv3`: `Conv2d(32, 3 * 4^2 = 48, kernel_size=3, padding=1)`
  - `pixel_shuffle`: `nn.PixelShuffle(4)`
- **Output range:** `conv3` has no final activation (e.g., `sigmoid` or `clamp`), so outputs are unbounded; during inference (`evaluate.py:95`, `server.py:125`), outputs are multiplied by `255.0` and clipped to `[0, 255]`.
- **Weight initialization:** Default PyTorch initialization (no custom weight initialization).

### 2.3 Training Data Generation (`train_sr.py`)
- `DIORPatchDataset` loads all 200 training images into RAM (`self.images`, ~384 MB for 200 × 800×800×3 uint8).
- `__len__` returns `len(self.images) * patches_per_image` (default `50`, i.e., 10,000 patches/epoch).
- `__getitem__` extracts a random `64×64` HR patch (`np.random.randint(0, h - 64)`), degrades it on-the-fly via `degrade_image(..., scale_factor=4, noise_sigma=15, blur_kernel=5, apply_haze=True)` to `16×16` LR, and normalizes both to `[0, 1]` float32 tensors.

### 2.4 Dataset Split Methodology (`prepare_dataset.py`)
- `extract_subset()` samples `TOTAL_IMAGES = 300` images from `JPEGImages-test.zip` using `random.seed(42)`.
- `split_dataset()` sorts `ORIGINAL_DIR`, sets `random.seed(42)`, shuffles, and splits into `train_images.txt` (first 200) and `eval_images.txt` (remaining 100).
- **No validation split:** There is only a 200-image train set and a 100-image eval set; no separate validation set exists for hyperparameter tuning or early stopping.

### 2.5 Degradation Parameters (`degrade.py`)
- `degrade_image(image, scale_factor=4, noise_sigma=15, blur_kernel=5, haze_t=0.85, haze_A=240, apply_haze=True)`
- Order of operations:
  1. **Downscale:** `cv2.resize` to `(w // scale_factor, h // scale_factor)` with `cv2.INTER_CUBIC`.
  2. **Haze:** `img_float * t + A * (1 - t)` (uniform scalar `t=0.85`, `A=240`), clipped to `[0, 255]` uint8.
  3. **Blur:** `cv2.GaussianBlur(lr, (blur_kernel, blur_kernel), 0)` (applied in **LR space**, not HR space).
  4. **Noise:** `np.random.normal(0, noise_sigma, lr.shape)` added to LR uint8, clipped to `[0, 255]` uint8.
  5. **Bicubic baseline:** `cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)`.

### 2.6 Determinism of Degradation
- **Non-deterministic:** `np.random.normal(0, noise_sigma, lr.shape)` uses global NumPy RNG state without a seed. Calling `degrade_image` twice on the same image produces different noise realizations.
- In `evaluate.py`, `generate_eval_data` runs without seeding `np.random`, so evaluation datasets are not strictly reproducible across runs.

### 2.7 Training Objective and Loss (`train_sr.py`)
- `criterion = nn.L1Loss()` computed in `[0, 1]` RGB space between `espcn(lr)` and `hr`.
- No perceptual loss, SSIM loss, or task-aware detection loss.

### 2.8 Optimizer and Scheduler (`train_sr.py`)
- `optimizer = optim.Adam(model.parameters(), lr=0.001)`
- **No learning rate scheduler** (constant `lr=0.001` for `epochs=15`, `batch_size=32`).

### 2.9 Data Augmentation (`train_sr.py`)
- **None** beyond random `64×64` spatial cropping. No random flips or 90° rotations (which are standard and label-preserving for aerial imagery).

### 2.10 Validation Pipeline
- **Absent during SR training:** `train_sr.py` has no validation loop, no validation PSNR/SSIM tracking, and no best-checkpoint saving; it only saves `weights/espcn_dior.pt` after the final epoch.

### 2.11 Object Detection Evaluation (`evaluate.py`)
- Generates three image directories under `data/eval/`: `original`, `degraded` (actually `bicubic_img`, i.e., bicubic-upscaled LR), and `sr_enhanced` (ESPCN output).
- Runs `model.val(data=yaml_path, imgsz=800, split='val', device='cpu', verbose=False, plots=False)`.
- Extracts `mAP50`, `mAP50_95`, `precision`, and `recall` from `results.results_dict` and saves to `data/benchmark_results.json`.

### 2.12 YOLO Weight Configuration
- `weights/yolov8_dior.pt` (~6.29 MB) is present in the repository.
- According to `prepare_dataset.py:54`, it was extracted from `Object_Detection_Satellite_Imagery_Yolov8_DIOR-main/runs/detect/yolov8n_epochs50_batch16/weights/best.pt` (a YOLOv8n model trained for 50 epochs, batch size 16 on DIOR).

### 2.13 Detector Weight Consistency
- `weights/yolov8_dior.pt` is fixed (frozen, inference-only) across all modalities in `evaluate.py` and `server.py`.

### 2.14 Potential Data Leakage
- **Train/Eval split:** `prepare_dataset.py` splits at the image level (200 train, 100 eval) from `JPEGImages-test.zip`, so there is no patch-level leakage between `train_images.txt` and `eval_images.txt`.
- **YOLO detector leakage:** `yolov8_dior.pt` was trained externally on DIOR (`yolov8n_epochs50_batch16`). Since `prepare_dataset.py` samples from `JPEGImages-test.zip`, it is unlikely to overlap with DIOR's official train split, though exact provenance of `yolov8_dior.pt`'s training split is unverified.

### 2.15 Potential Issues in Synthetic Degradation Modeling (`degrade.py`)
1. **Blur in LR space:** Applying a `5×5` Gaussian blur *after* 4× downsampling corresponds to an unrealistically large `~20×20` blur kernel in HR space, severely destroying small objects.
2. **Haze before blur/noise:** Atmospheric scattering is applied in LR space before sensor blur/noise, with a single uniform scalar `t=0.85` (ignoring depth/spatial variation).
3. **Quantization:** Intermediate `uint8` clipping after haze and after noise introduces repeated rounding/quantization errors.
4. **Hardcoded parameters:** `train_sr.py` and `evaluate.py` hardcode `scale_factor=4, noise_sigma=15, blur_kernel=5, apply_haze=True` without config files.
5. **Misleading naming in `evaluate.py`:** `dirs['degraded']` stores `bicubic_img` (upscaled), not the raw LR image.

### 2.16 Missing Metrics
- **Image restoration metrics:** No PSNR, SSIM, or LPIPS anywhere in `train_sr.py` or `evaluate.py`.
- **Detection metrics:** No per-class AP, no small/medium/large object breakdown, no false-positive / false-negative counts.
- **Efficiency metrics:** No parameter count, FLOPs, latency, throughput, or peak memory tracking.

### 2.17 Existing Limitations
- `train_sr.py:80`, `evaluate.py:57,139`, and `server.py:29` hardcode `device = torch.device('cpu')` (no GPU support).
- `prepare_dataset.py:18` uses Windows-style path `os.path.expanduser('~\\Downloads')`.
- `train_sr.py` and `evaluate.py` have no CLI arguments or config management.
- `server.py` has no error handling if `espcn_model` is `None` or weights are missing (`process_image` will crash on `espcn_model(lr_tensor)`).

### 2.18 Reproducibility Issues
- No `requirements.txt` or `pyproject.toml` pinning package versions.
- No global seeding for `torch`, `numpy`, or `random` in `train_sr.py` or `evaluate.py`.
- `np.random.randint` and `np.random.normal` in `DIORPatchDataset.__getitem__` are unseeded.
- `num_workers=0` in `DataLoader` is deterministic in worker order, but `shuffle=True` is unseeded.

### 2.19 Runtime and Memory Bottlenecks
- `DIORPatchDataset` loads all 200 HR images into RAM upfront; scaling to full DIOR (thousands of images) would exhaust memory.
- `num_workers=0` performs on-the-fly OpenCV degradation sequentially on the main thread.
- CPU-only training and evaluation (`device='cpu'`) is a major bottleneck.

### 2.20 Inherited vs. Newly Developed Functionality
- **Inherited / Adapted:**
  - `weights/yolov8_dior.pt` (extracted from `Object_Detection_Satellite_Imagery_Yolov8_DIOR-main`).
  - Standard ESPCN architecture (`sr_model.py`), basic DIOR XML-to-YOLO conversion (`prepare_dataset.py`), synthetic degradation (`degrade.py`), training loop (`train_sr.py`), evaluation (`evaluate.py`), and FastAPI/HTML dashboard (`server.py`, `frontend/`).
- **Newly Developed (Current Audit):**
  - `docs/PROJECT_AUDIT.md` (this audit document). All existing code remains unmodified at Stage 1.
