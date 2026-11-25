# AeroSR: Degradation-Aware Aerial Image Restoration & Small-Object Detection

**AeroSR** investigates degradation-aware aerial image restoration and its impact on downstream small-object detection using a fixed YOLOv8n detector on the DIOR dataset.

> **Attribution & Provenance:** The initial repository (`degrade.py`, `sr_model.py`, `train_sr.py`, `evaluate.py`, `prepare_dataset.py`, `server.py`, `frontend/`, and `weights/yolov8_dior.pt`) was adapted from existing work. All modular research infrastructure under `src/aerosr/`, `configs/`, `scripts/`, `tests/`, and `docs/` represents original engineering and experimental contributions. See [PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md) for a full breakdown.

---

## Research Questions
1. Under different image degradation conditions (downsampling, blur, noise, haze, compression), which degradation causes the largest loss in aerial object detection?
2. Which lightweight super-resolution architectures (Bicubic, ESPCN, SRCNN, FSRCNN, ResidualSRNet) best recover downstream detection accuracy, and at what computational cost?
3. Do pixel-wise image quality metrics (PSNR, SSIM) correlate with downstream object detection (`mAP@0.5`)?
4. How well do restoration models generalize to unseen degradation distributions?

---

## Key Results (DIOR 100-Image Held-Out Test Set)
All results are generated from `outputs/all_results.json` using fixed `weights/yolov8_dior.pt` (`seed=42`).

| Method | Params | Latency (ms) | PSNR (dB) | SSIM | mAP@0.5 | Small-Object Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (Reference)** | — | — | inf | 1.0000 | **0.7543** | **0.7179** |
| **Bicubic (Degraded)** | — | — | 19.10 | 0.4208 | 0.1863 | 0.0959 |
| **FSRCNN** | 24,683 | 73.65 | 22.52 | 0.5197 | 0.1966 | 0.0310 |
| **SRCNN** | 69,251 | 952.67 | 24.41 | 0.5821 | 0.3247 | 0.1580 |
| **ESPCN (Baseline)** | 37,200 | **47.35** | 24.23 | 0.5633 | 0.4071 | 0.1594 |
| **ResidualSRNet (Ours)** | 88,752 | 104.49 | **24.96** | **0.6081** | **0.5193** | **0.3907** |

See [RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) for full results across Experiments A–F and generated figures in `outputs/figures/`.

---

## Repository Structure
```text
AeroSR/
├── configs/                 # YAML degradation & experiment configs
├── docs/                    # Audit, architecture, protocol, and research report
├── src/aerosr/              # Modular PyTorch package (degradation, models, training, evaluation, plots)
├── scripts/                 # Split preparation & master experiment runner
├── tests/                   # Pytest unit tests
├── weights/                 # Fixed YOLOv8n DIOR weights
└── pyproject.toml           # Package & test configuration
```

---

## Quick Start
```bash
python3 -m venv .venv
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest -v
.venv/bin/python scripts/prepare_splits.py
.venv/bin/python scripts/run_experiments.py --epochs 5 --patches-per-image 20 --seed 42
```
