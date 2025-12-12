# 🏥 Anatomy-Aware CT Scan Denoising

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Advanced Deep Learning Approaches for Radiation Dose Reduction in Medical Imaging**

This repository contains a comprehensive collection of state-of-the-art deep learning models for CT scan denoising with anatomy-aware processing. Our approach enables significant radiation dose reduction while maintaining diagnostic image quality.

---

## 🎯 Project Overview

Low-dose CT imaging is crucial for reducing radiation exposure in patients, but it introduces noise and artifacts that degrade image quality. This project implements and compares multiple deep learning architectures to perform anatomy-aware denoising on low-dose CT scans across different radiation dose levels.

### Key Features

✅ **Multi-Model Architecture**: Compare 6 different deep learning models  
✅ **Multi-Dose Evaluation**: Evaluate performance across 4 radiation dose levels (10%, 25%, 50%, 70%)  
✅ **Anatomy-Aware Processing**: Models trained with knowledge of anatomical structures  
✅ **Comprehensive Metrics**: PSNR, SSIM, RMSE evaluation  
✅ **Teacher-Student Training**: Knowledge distillation approach for improved performance  

---

## 📊 Repository Structure

```
Anatomy-aware-CT-scan-denoising/
├── Baseline/                 # Basic Autoencoder Implementation
│   ├── training_baseline.py  # Training script
│   ├── evaluate_baseline.py  # Evaluation script
│   └── eval_images/          # Sample outputs
│
├── Nafnet/                   # NAFNet (Normalized Attention FNet)
│   ├── train_nafnet_mlp.py   # MLP variant training
│   ├── evaluate_nafnet.py    # Evaluation
│   └── eval_images_nafnet_mlp/
│
├── RadIMG+Nafnet/            # RAD-IMG enhanced NAFNet
│   ├── train_nafnet_radimg.py
│   ├── evaluate_rad.py
│   └── eval_images_rad_mlp/
│
├── Resnet/                   # ResNet-based Architecture
│   ├── train_resnet.py
│   ├── evaluate_resnet.py
│   └── eval_images_resnet/
│
├── unet/                     # U-Net Architecture
│   ├── train_unet.py
│   ├── evaluate_unet.py
│   └── eval_images_unet/
│
├── Wo_dose/                  # Ablation Study (Without Dose)
│   ├── train_wodose.py
│   ├── evaluate_no_dose.py
│   └── eval_images_ablation/
│
├── Noise Simulation/         # Data Preparation
│   ├── data_LoD0.py          # Low-dose simulation
│   └── data_mayo.py          # Mayo clinic data processing
│
├── Results/                  # Model Comparison Results
│   ├── model_comparisons.md
│   ├── metrics_summary.csv
│   └── visualizations/
│
├── Presentation - Anatomy-Aware Denoising.pdf
└── README.md
```

---

## 🧠 Implemented Models

| Model | Architecture | Parameters | Focus Area |
|-------|-------------|-----------|------------|
| **Baseline** | Autoencoder | Conv + Deconv | Foundation model |
| **NAFNet** | Normalized Attention FNet | Attention-based | Feature refinement |
| **RAD-IMG + NAFNet** | NAFNet + RadIMG | Enhanced attention | Anatomy-aware processing |
| **ResNet** | Residual Networks | Skip connections | Deep feature learning |
| **U-Net** | Encoder-Decoder | Dense connections | Semantic segmentation-style |
| **Wo_Dose** | No dose conditioning | Ablation baseline | Performance impact analysis |

---

## 📈 Performance Metrics

Our models are evaluated on the following metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (typical range: 20-40 dB)
- **SSIM (Structural Similarity Index)**: Range [0,1], higher indicates better structural preservation
- **RMSE (Root Mean Square Error)**: Lower is better
- **MSE (Mean Square Error)**: Pixel-level error measurement

### Evaluation by Radiation Dose

Models are tested across 4 dose levels:
- **10% Dose**: Extreme noise reduction scenario
- **25% Dose**: Challenging noise environment
- **50% Dose**: Moderate dose level
- **70% Dose**: Near-standard dose

---

## 🚀 Training Pipeline

### Model Training

The teacher network is trained on normal-dose CT (NDCT) images:

```python
python Baseline/training_baseline.py \
    --mayo_root /path/to/data \
    --epochs_teacher 100 \
    --batch 8 \
    --lr 2e-4
    --epochs_student 150 \
    --lam_lat 1.0 \
    --lam_rec 1.0
```



## 📥 Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9+
- NumPy, Pillow, tqdm
- TensorBoard for visualization

### Setup

```bash
# Clone the repository
git clone https://github.com/Ajay02423/Anatomy-aware-CT-scan-denoising.git
cd Anatomy-aware-CT-scan-denoising

# Install dependencies
pip install torch torchvision
pip install numpy pillow tqdm tensorboard
```

---

## 🎓 Training Configuration

Key hyperparameters used across models:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 8 | Samples per iteration |
| Learning Rate | 2e-4 | Adam optimizer |
| Teacher Epochs | 100 | NDCT autoencoder training |
| Student Epochs | 150 | LDCT encoder training |
| α_SSIM | 0.2 | SSIM loss weight |
| λ_lat | 1.0 | Latent space loss weight |
| λ_rec | 1.0 | Reconstruction loss weight |

---

## 📊 Normalization Strategy

Images are normalized using fixed windowing:

```python
MIN_HU = -1000.0  # Minimum Hounsfield Unit
MAX_HU = 1000.0   # Maximum Hounsfield Unit
# Normalize to [0, 1] then to [-1, 1] for Tanh activation
```

---

## 🔍 Loss Functions

### Teacher Training
```
Loss = L1_Loss + α_SSIM × (1 - SSIM)
```

### Student Training
```
Loss = λ_lat × MSE_Latent + λ_rec × L1_Reconstruction
```

---

## 📝 Dataset Information

The project uses:
- **LDCT Pairs Dataset**: Low-dose and Normal-dose CT scan pairs
- **Doses**: 10%, 25%, 50%, 70% of standard radiation
- **Format**: .npy files with Hounsfield Unit values
- **Normalization**: Per-sample HU windowing

---

## 🎨 Results & Visualization

Sample outputs are saved in each model folder:
- `training_samples/`: Progressive training visualization
- `eval_images/`: Model evaluation results
- `dose_wise_results/`: Performance per radiation dose

---

## 📚 Model Training Details

### Baseline Encoder Architecture
```
┌─────────────────────────────────────┐
│ Input (1, H, W)                     │
├─────────────────────────────────────┤
│ Conv Block (1 → 64)                 │
│ MaxPool → Conv Block (64 → 128)     │
│ MaxPool → Conv Block (128 → 256)    │
│ MaxPool → Conv Block (256 → 512)    │
│ MaxPool → Conv Block (512 → 512)    │
└─────────────────────────────────────┘
         ↓ Latent Space
```

### Baseline Decoder Architecture
```
┌─────────────────────────────────────┐
│ Latent (512, h, w)                  │
├─────────────────────────────────────┤
│ DeconvBlock (512 → 512)             │
│ ConvBlock → DeconvBlock (512 → 256) │
│ ConvBlock → DeconvBlock (256 → 128) │
│ ConvBlock → DeconvBlock (128 → 64)  │
│ Conv (64 → 1) + Tanh                │
└─────────────────────────────────────┘
```

---

## 🎯 Key Findings

1. **Dose Dependency**: Model performance scales with radiation dose
2. **Anatomy Awareness**: RAD-IMG enhancement shows consistent improvements
3. **Trade-offs**: Balance between noise reduction and detail preservation
4. **Generalization**: Models trained on one dose generalize reasonably to others

---

## 📖 How to Evaluate

```bash
# Evaluate Baseline model
python Baseline/evaluate_baseline.py \
    --model_path runs/final/student \
    --test_data /path/to/test

# Evaluate NAFNet
python Nafnet/evaluate_nafnet.py \
    --model_path runs/final/student \
    --test_data /path/to/test
```

---

## 🔄 Knowledge Distillation Approach

Our training uses a novel teacher-student framework:

1. **Teacher Network**: Learns to denoise normal-dose images
2. **Student Network**: Learns from teacher's latent representations
3. **Knowledge Transfer**: Minimize latent space divergence
4. **Dose Agnostic**: Student works across all dose levels

---

</div>
