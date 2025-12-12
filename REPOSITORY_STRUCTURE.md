# 📂 Repository Structure Guide

## Overview

This document provides a comprehensive guide to the repository organization, helping users understand where to find training scripts, evaluation code, data processing utilities, and results.

---

## 📁 Main Directory Structure

```
Anatomy-aware-CT-scan-denoising/
├── 📁 Baseline/                    # Foundation autoencoder model
├── 📁 Nafnet/                      # NAFNet model with MLP variant
├── 📁 RadIMG+Nafnet/               # Anatomy-aware NAFNet (RAD-IMG enhanced)
├── 📁 Resnet/                      # ResNet-based architecture
├── 📁 unet/                        # U-Net encoder-decoder model
├── 📁 Wo_dose/                     # Ablation study (no dose conditioning)
├── 📁 Noise Simulation/            # Data preparation and augmentation
├── 📁 Results/                     # Performance metrics and comparisons
├── 📄 README.md                    # Main project documentation
├── 📄 REPOSITORY_STRUCTURE.md      # This file
└── 📄 Presentation - Anatomy-Aware Denoising.pdf  # Project presentation
```

---

## 🧠 Model Directories

Each model directory follows a consistent structure:

### Baseline/ 
**Basic Autoencoder Implementation**
```
Baseline/
├── training_baseline.py           # Teacher-student training pipeline
├── evaluate_baseline.py           # Evaluation metrics computation
├── eval_images/                   # Sample reconstruction outputs
│   └── Screenshot_2025-12-07 032427.png
└── (model checkpoints saved during training)
```

**Key Files:**
- `training_baseline.py`: Implements both teacher network (NDCT autoencoder) and student network (LDCT encoder) with knowledge distillation
- `evaluate_baseline.py`: Computes PSNR, SSIM, RMSE metrics per-dose level

### Nafnet/
**NAFNet with MLP Variant**
```
Nafnet/
├── train_nafnet_mlp.py            # Training script for MLP-based NAFNet
├── evaluate_nafnet.py             # Evaluation script
├── eval_images_nafnet_mlp/        # Output samples
└── (model checkpoints)
```

**Key Files:**
- `train_nafnet_mlp.py`: Implements normalized attention-based feature extraction
- `evaluate_nafnet.py`: Per-dose evaluation

### RadIMG+Nafnet/
**Anatomy-Aware NAFNet (RAD-IMG Enhanced)**
```
RadIMG+Nafnet/
├── train_nafnet_radimg.py         # RAD-IMG enhanced training
├── evaluate_rad.py                # Radiation-dose aware evaluation
├── eval_images_rad_mlp/           # Anatomy-preserving outputs
└── (model checkpoints)
```

**Key Files:**
- `train_nafnet_radimg.py`: Incorporates anatomy-aware features via RAD-IMG module
- `evaluate_rad.py`: Specialized evaluation for anatomy-aware metrics

### Resnet/
**ResNet-Based Architecture**
```
Resnet/
├── train_resnet.py                # ResNet training
├── evaluate_resnet.py             # Evaluation
├── eval_images_resnet/            # Output samples
└── (model checkpoints)
```

### unet/
**U-Net Encoder-Decoder**
```
unet/
├── train_unet.py                  # U-Net training
├── evaluate_unet.py               # Evaluation
├── eval_images_unet/              # Output samples
└── (model checkpoints)
```

### Wo_dose/
**Ablation Study (Without Dose Conditioning)**
```
Wo_dose/
├── train_wodose.py                # Training without dose awareness
├── evaluate_no_dose.py            # Evaluation
├── eval_images_ablation/          # Ablation outputs
└── (model checkpoints)
```

**Purpose:** Demonstrates the importance of dose-aware training

---

## 🔧 Noise Simulation Directory

```
Noise Simulation/
├── data_LoD0.py                   # Low-dose CT simulation from normal-dose
└── data_mayo.py                   # Mayo Clinic LDCT dataset processing
```

**Purpose:**
- `data_LoD0.py`: Simulates low-dose images by adding Poisson noise
- `data_mayo.py`: Loads and preprocesses real LDCT pairs from Mayo dataset

**Usage:**
```python
# Prepare training data
python Noise_Simulation/data_mayo.py --input /path/to/data --output ./processed_data
```

---

## 📊 Results Directory

```
Results/
├── model_comparison.md            # Detailed performance comparison tables
└── metrics_summary.csv            # Machine-readable metrics data
```

### model_comparison.md
Contains:
- Overall performance metrics table (PSNR, SSIM, RMSE averages)
- Dose-wise performance (10%, 25%, 50%, 70% doses)
- Key findings and insights
- Model recommendations
- Visualization references

### metrics_summary.csv
Structured CSV with:
- Model performance summary
- Dose-level PSNR comparison
- Dose-level SSIM comparison
- Dose-level RMSE comparison
- Training configuration details
- Metric definitions

**Usage:**
```python
import pandas as pd
metrics = pd.read_csv('Results/metrics_summary.csv')
print(metrics.head())
```

---

## 🚀 Training Workflow

### Step 1: Data Preparation
```bash
cd Noise_Simulation/
python data_mayo.py --root /DATA/CT/LDCT_pairs/Mayo_pairs/1mm_B30 --output /processed_data
```

### Step 2: Train Model
```bash
cd ../Baseline/  # or Nafnet/, RadIMG+Nafnet/, etc.
python training_baseline.py \
    --mayo_root /processed_data \
    --epochs_teacher 100 \
    --epochs_student 150 \
    --batch 8 \
    --out runs/final
```

### Step 3: Evaluate Model
```bash
python evaluate_baseline.py \
    --model_path runs/final/student/student_Es.pt \
    --test_data /processed_data/test \
    --output Results/evaluation
```

---

## 📈 Key Files Across Models

| File | Purpose | Models |
|------|---------|--------|
| `training_*.py` | Main training script | All |
| `evaluate_*.py` | Evaluation and metrics | All |
| `eval_images/` | Visual output samples | All |
| Model checkpoints | Saved weights (.pt) | All |

---

## 💾 Expected Directory Tree After Training

```
Baseline/
├── runs/final/
│   ├── teacher/
│   │   ├── teacher_Ec.pt          # Teacher encoder weights
│   │   ├── decoder_D.pt           # Shared decoder weights
│   │   ├── tensorboard/           # TensorBoard logs
│   │   └── samples/               # Training samples
│   ├── student/
│   │   ├── student_Es.pt          # Student encoder weights
│   │   ├── tensorboard/
│   │   └── samples/
│   └── eval/
│       └── dose_*_example.png     # Evaluation samples per dose
```

---

## 🎯 File Naming Conventions

### Training Scripts
- `train_*.py`: Teacher-student training
- Format: `train_{architecture}.py` or `train_{architecture}_{variant}.py`

### Evaluation Scripts
- `evaluate_*.py`: Model evaluation
- Format: `evaluate_{architecture}.py` or `evaluate_{variant}.py`

### Output Images
- `Screenshot_YYYY-MM-DD HHMMSS.png`: Sample outputs
- `dose_{percentage}_example.png`: Per-dose examples

### Model Checkpoints
- `teacher_Ec.pt`: Teacher encoder
- `student_Es.pt`: Student encoder
- `decoder_D.pt`: Shared decoder

---

## 📝 Configuration Files

Each model uses command-line arguments for configuration (no separate config files).

**Common Arguments:**
```
--mayo_root          Path to LDCT dataset root
--out                Output directory for checkpoints and logs
--epochs_teacher     Number of teacher training epochs (default: 100)
--epochs_student     Number of student training epochs (default: 150)
--batch              Batch size (default: 8)
--lr                 Learning rate (default: 2e-4)
--alpha_ssim         SSIM loss weight (default: 0.2)
--lam_lat            Latent space loss weight (default: 1.0)
--lam_rec            Reconstruction loss weight (default: 1.0)
```

---

## 📖 Documentation Files

| File | Content |
|------|----------|
| README.md | Project overview, installation, training guide |
| REPOSITORY_STRUCTURE.md | This file - directory organization |
| Results/model_comparison.md | Performance comparison and analysis |
| Results/metrics_summary.csv | Structured metrics data |
| Presentation - Anatomy-Aware Denoising.pdf | Project presentation slides |

---

## 🔄 Relationships Between Components

```
Data Preparation (Noise Simulation/)
         ↓
    Training (Each Model/)
         ↓
  Shared Decoder
         ↓
  Evaluation (evaluate_*.py)
         ↓
   Results Analysis
         ↓
Results/ (Comparisons & Metrics)
```

---

## 💡 Tips for Navigation

1. **Starting with a model:** Check `training_*.py` for implementation details
2. **Understanding performance:** See `Results/model_comparison.md`
3. **Comparing metrics:** Open `Results/metrics_summary.csv` in a spreadsheet
4. **Visualizing results:** Check `eval_images/` in each model folder
5. **Reproducing results:** Follow the workflow in README.md

---

## 🆘 Troubleshooting

**Models not training:**
- Ensure LDCT dataset is in correct format (.npy files)
- Verify HU values are in [-1000, 1000] range
- Check GPU availability with `torch.cuda.is_available()`

**Missing evaluation images:**
- Training must complete at least 1 epoch
- Check output directory has write permissions
- Verify sufficient disk space for tensorboard logs

**Metric discrepancies:**
- Different SSIM window sizes may give different values
- Ensure test set is consistent across models
- Check data normalization matches training set

---

**Last Updated:** December 12, 2025
**Created by:** Ajay02423
**License:** MIT
