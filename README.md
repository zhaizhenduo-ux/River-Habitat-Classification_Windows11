# River-Habitat-Classification-Windows11


This repository provides an end-to-end deep learning pipeline for **river habitat segmentation** and **debris detection** using high-resolution UAV orthomosaic imagery. The workflow integrates:

- ✅ Semantic segmentation using DeepLab (FCN + VGG16)
- ✅ Domain knowledge–based spatial refinement
- ✅ Segment Anything Model (SAM) for further mask enhancement
- ✅ Detectron2-based debris detection
- ✅ Tiling and merging utilities to handle large `.tif` orthomosaics

> 📌 This software runs on Windows via WSL (Ubuntu) and supports GPU acceleration.

---

## 📦 Features

- 🌍 Process ultra-large UAV `.tif` orthomosaics with split/merge support
- 🎯 Multi-stage habitat classification: segmentation → correction → refinement
- 🤖 Object detection for floating debris using Detectron2
- 💻 Designed for both local development and headless server execution
- 🚀 GPU support for all model inference steps (CUDA-compatible)

---

## 🖥️ System Requirements

| Component       | Requirement                  |
|----------------|------------------------------|
| OS             | Windows 10 / 11 with WSL     |
| WSL Linux      | Ubuntu (via `wsl --install`) |
| GPU            | NVIDIA GPU w/ CUDA ≥ 11.0    |
| Python         | Version 3.8 (recommended)    |
| RAM            | ≥ 16 GB (8 GB minimum)       |

---

## ⚙️ Installation Guide (Windows)

### 1. Install WSL & Ubuntu
```powershell
wsl --install
```
Reboot to complete WSL setup.

### 2. Install Anaconda in WSL
Download and run:
```bash
bash Anaconda3-202x.sh
```
After installation, restart the terminal.

### 3. Create Conda Environment
```bash
conda create --name Segment python=3.8
conda activate Segment
```

### 4. Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 5. Install TensorFlow and Dependencies
```bash
pip install tensorflow tensorboard boto3 cryptography
```

### 6. Install Required Python Packages
```bash
pip install opencv-python cchardet pandas scipy pyexiv2 rasterio tqdm psutil scikit-image matplotlib
```

### 7. Install Segment Anything Model (SAM)
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 8. Install Detectron2
```bash
sudo apt-get update
sudo apt-get install -y gcc g++ make cmake libopenblas-dev libomp-dev
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## 📁 Dataset Format

Place input files into a folder such as `test_data/`:
```
test_data/
├── P4_10_2_2023_transparent_mosaic_group1.tif       # Required: orthomosaic
├── P4_10_2_2023_transparent_mosaic_group1.jpg       # Optional: visual
├── P4_10_2_2023_transparent_mosaic_group1.json      # Optional: ground truth
```

---

## ▶️ How to Run

```bash
python final_pipeline.py   --image_path ./test_data/P4_10_2_2023_transparent_mosaic_group1.tif   --output_folder_path ./test_data/result/
```

The pipeline performs:

1. Image tiling and downscaling
2. FCN segmentation
3. Domain adjustment
4. SAM refinement
5. Debris detection
6. Re-merging into full-size outputs

---

## 📤 Output Folder Structure

```
result/
├── step1_result/         # Habitat segmentation (FCN)
├── step2_result/         # Domain-adjusted refinement
├── step3_result/         # SAM-enhanced refinement
├── step4_result/         # Debris detection
└── merged_downscaled/    # Final downscaled merged outputs
```

---

## 📌 Notes & Recommendations

- For large orthomosaics (e.g., >10,000×10,000 px), use a GPU with at least 8 GB VRAM.
- Run on headless servers with CUDA support for best performance.
- `.json` files are optional and only needed for evaluation.
- The output is structured to allow GIS visualization (downscaled `.jpg` and `.png` maps).

---

## 📸 Sample Results

| Stage         | Output                             |
|---------------|------------------------------------|
| Step 1        | `color_mask.jpg`                   |
| Step 2        | `processed_mask_domain_adjust.png` |
| Step 3        | `predict_step3.png`                |
| Step 4        | `debris_detection.jpg`             |

---

## 🧠 Acknowledgements

- [Facebook Research – SAM](https://github.com/facebookresearch/segment-anything)
- [Detectron2](https://github.com/facebookresearch/detectron2)

---



## 🙋 Contact

Maintained by: **[Zhenduo Zhai]**  
Lab: **Intelligent & Distributed Computing Lab, University of Missouri**  
Email: `zz7z9@missouri.edu`
