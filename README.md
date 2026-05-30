# AeroSR: Aerial Super-Resolution for Enhanced Object Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)

AeroSR is a lightweight, end-to-end **Super-Resolution (SR) preprocessing pipeline** designed to restore synthetically degraded aerial and satellite imagery, significantly improving downstream object detection accuracy. 

Aerial images captured by drones and satellites often suffer from atmospheric haze, sensor noise, and motion blur. This project demonstrates that enhancing these low-resolution, degraded images using an **Efficient Sub-Pixel Convolutional Neural Network (ESPCN)** leads to quantifiable improvements in object detection performance using state-of-the-art models like YOLOv8.

---

## 🔬 Methodology

Our approach integrates synthetic image degradation with a deep-learning-based restoration pipeline:

1. **Synthetic Degradation Modeling (`degrade.py`)**: 
   High-resolution ground truth images are mathematically degraded to simulate real-world aerial conditions. The degradation model includes:
   - **4x Downsampling** (Bicubic interpolation)
   - **Atmospheric Haze** (Atmospheric scattering model)
   - **Gaussian Blur** (Simulating motion/lens blur)
   - **Gaussian Noise** (Simulating sensor noise)

2. **Super-Resolution Model (`sr_model.py`)**:
   We implement the **ESPCN** architecture, a highly efficient PyTorch model designed for real-time super-resolution. The model learns to reconstruct high-frequency details from the degraded low-resolution patches by utilizing a sub-pixel convolution layer, making it computationally lightweight and suitable for edge-device deployment.

3. **Object Detection (`server.py`)**:
   We utilize **YOLOv8** to benchmark the effectiveness of the SR reconstruction. The pipeline compares YOLO bounding box predictions across four modalities:
   - **Original (Ground Truth)**
   - **Degraded (Low-Resolution)**
   - **Bicubic Upscaled (Baseline)**
   - **ESPCN Super-Resolved (AeroSR)**

---

## 📂 Repository Structure

```text
AeroSR/
├── data/                  # Datasets (e.g., DIOR dataset) - *Not tracked by git*
├── frontend/              # HTML/JS/CSS for the visual benchmark dashboard
├── weights/               # PyTorch model weights (ESPCN, YOLOv8)
├── degrade.py             # Synthetic degradation model (haze, blur, noise)
├── evaluate.py            # Script for quantitative benchmarking (mAP evaluation)
├── prepare_dataset.py     # Utility to download and slice the DIOR dataset
├── server.py              # FastAPI inference backend & UI host
├── sr_model.py            # ESPCN network architecture
├── train_sr.py            # PyTorch training script for the ESPCN model
└── test_download.py       # Dataset download utility
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CPU or NVIDIA GPU + CUDA Toolkit

### Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/YusufMalu001/AeroSR.git
cd AeroSR
pip install torch torchvision opencv-python ultralytics fastapi uvicorn pydantic numpy tqdm
```

---

## 🛠️ Usage

### 1. Training the Super-Resolution Model
To train the ESPCN model on the degraded image patches against the original high-resolution ground truths:
```bash
python train_sr.py
```
*Note: Ensure `prepare_dataset.py` has been executed to generate the required training dataset slices.*

### 2. Running the Inference Dashboard
AeroSR includes a fully interactive web dashboard to visually compare the YOLOv8 detections on degraded vs. super-resolved images.

Start the FastAPI server:
```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```
Then navigate to `http://127.0.0.1:8000/static/index.html` in your web browser.

---

## 📊 Evaluation & Results
You can evaluate the quantitative object detection performance (mAP50) using the `evaluate.py` script. The results will demonstrate how Super-Resolution preprocessing recovers detection fidelity lost to atmospheric and sensor degradation.

---

## 📜 License
This project is open-source and available under the MIT License.
