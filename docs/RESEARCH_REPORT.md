# AeroSR Research Report

**Generated from:** `outputs/all_results.json`  
**Dataset:** DIOR (160 train / 40 val / 100 test images, 949 test instances, `seed=42`)  
**Fixed Detector:** `weights/yolov8_dior.pt` (YOLOv8n, `imgsz=800`, CPU)

---

## 1. Executive Summary
Under combined synthetic aerial degradation (4× bicubic downsampling, 5×5 HR Gaussian blur, $\sigma=15$ Gaussian noise, and $t=0.85$ atmospheric haze), YOLOv8n detection performance drops from **0.7543** `mAP@0.5` (Clean) to **0.1863** (Bicubic). Training a lightweight **ResidualSRNet** (88.8K params) on the 160-image training split recovers `mAP@0.5` to **0.5193** (+0.3330 over Bicubic) and small-object recall from **0.0959** to **0.3907**.

---

## 2. Experiment A: Degradation Sensitivity
| Condition | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | PSNR (dB) | SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | 0.7543 | 0.5453 | 0.8237 | 0.6645 | inf | 1.0000 |
| `blur_only` (k=7) | 0.6947 | 0.4952 | 0.7941 | 0.6164 | 28.62 | 0.8112 |
| `haze_only` (t=0.65) | 0.6908 | 0.4994 | 0.7876 | 0.6072 | 13.94 | 0.8216 |
| `downsample_only` (4×) | 0.5478 | 0.3751 | 0.7064 | 0.5057 | 25.69 | 0.7151 |
| `noise_only` ($\sigma=25$) | 0.3432 | 0.2299 | 0.6665 | 0.3280 | 20.33 | 0.3527 |
| `downsample_blur` | 0.6299 | 0.4270 | 0.7592 | 0.5840 | 26.86 | 0.7340 |
| `blur_noise` | 0.4230 | 0.2901 | 0.7609 | 0.3649 | 23.45 | 0.4334 |
| `downsample_noise` | 0.2526 | 0.1570 | 0.7904 | 0.1938 | 22.42 | 0.4586 |
| `combined` | **0.1863** | **0.1138** | 0.1901 | 0.3060 | 19.10 | 0.4208 |

**Key Finding:** Additive Gaussian noise is the single most destructive degradation for YOLOv8n (`0.7543 -> 0.3432`), followed by 4× downsampling (`0.5478`). In the noise severity sweep ($\sigma \in \{0, 10, 20, 35\}$), `mAP@0.5` drops from `0.7543` to `0.6971`, `0.4132`, and `0.2397`.

---

## 3. Experiment B & F: Restoration Models & Efficiency
| Method | Params | Size (MB) | Latency (ms) | FPS | PSNR (dB) | SSIM | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | — | — | — | — | inf | 1.0000 | 0.7543 | 0.5453 |
| `bicubic` | — | — | — | — | 19.10 | 0.4208 | 0.1863 | 0.1138 |
| `espcn` | 37,200 | 0.142 | **47.35** | **21.12** | 24.23 | 0.5633 | 0.4071 | 0.2785 |
| `srcnn` | 69,251 | 0.264 | 952.67 | 1.05 | 24.41 | 0.5821 | 0.3247 | 0.2119 |
| `fsrcnn` | 24,683 | 0.094 | 73.65 | 13.58 | 22.52 | 0.5197 | 0.1966 | 0.1193 |
| `residual_sr` | 88,752 | 0.339 | 104.49 | 9.57 | **24.96** | **0.6081** | **0.5193** | **0.3460** |

**Experiment D (PSNR/SSIM vs. Detection):** `SRCNN` achieves higher PSNR (24.41 vs. 24.23 dB) and SSIM (0.5821 vs. 0.5633) than `ESPCN`, yet scores lower on `mAP@0.5` (0.3247 vs. 0.4071). Pixel-wise reconstruction metrics do not strictly predict downstream detection accuracy.

---

## 4. Experiment C: Cross-Degradation Generalization (`mAP@0.5`)
| Unseen Degradation | Bicubic | ESPCN | SRCNN | FSRCNN | ResidualSRNet |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `blur_mismatch_k9` | 0.1459 | 0.3856 | 0.2550 | 0.1914 | **0.4795** |
| `noise_mismatch_s30` | 0.0178 | **0.1428** | 0.0682 | 0.0858 | 0.1355 |
| `bilinear_downsample` | 0.1732 | 0.4028 | 0.3066 | 0.1984 | **0.5039** |
| `heavy_haze_t06` | 0.0682 | 0.2906 | 0.1635 | 0.1481 | **0.3298** |
| `jpeg_q50` | 0.2286 | 0.2977 | 0.2536 | 0.1728 | **0.3893** |

---

## 5. Experiment E: Small-Object Performance
In the 100-image test set, 709 of 949 objects (74.7%) are `small` (<0.5% image area).
- **Clean:** Small Recall = `0.7179` (68 FP), Medium = `0.8539`, Large = `0.9032`
- **Bicubic:** Small Recall = `0.0959` (0 FP), Medium = `0.3146`, Large = `0.0484`
- **ESPCN:** Small Recall = `0.1594` (1 FP), Medium = `0.5393`, Large = `0.4032`
- **ResidualSRNet:** Small Recall = **`0.3907`** (2 FP), Medium = **`0.6348`**, Large = **`0.5645`**

---

## 6. Limitations & Uncertainty
- All SR models were trained for 5 epochs on CPU (`160` train images, `20` patches/image) with L1 loss; longer GPU training or perceptual/task-aware losses would likely improve all models, particularly `FSRCNN`.
- `yolov8_dior.pt` is fixed and inherited; its original training split provenance is external to this repository.
