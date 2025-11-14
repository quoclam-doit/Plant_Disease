# Plant Disease Classification with MambaTSR

🌿 **Deep learning model for plant disease classification using MambaTSR (Vision Mamba) architecture**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Kết Quả

- **Model**: MambaTSR (VSSM-Tiny) - 77M parameters
- **Dataset**: PlantVillage - 54,304 images, 39 classes
- **Best Accuracy**: **98.96%** validation accuracy
- **Training Time**: 3 hours (RTX 5060 Ti 16GB)
- **Hardware**: RTX 5060 Ti (sm_120) with CUDA forward compatibility

## 📁 Cấu Trúc Project

```
Plant_Disease/
├── train_mambatsr_plantvillage.py  ← Main training script
├── generate_training_plots.py      ← Visualization utilities
├── setup_mambatsr.py               ← Setup script (auto-download)
│
├── docs/                           ← Documentation
│   ├── reports/                    → Technical reports
│   ├── guides/                     → Setup & training guides
│   └── presentations/              → Presentation materials
│
├── scripts/                        ← Utility scripts
│   ├── tests/                      → Test scripts
│   └── shell/                      → Shell scripts
│
├── notebooks/                      ← Jupyter notebooks
│   ├── Plant_Disease_EDA.ipynb
│   └── Plant_Disease_MambaTSR.ipynb
│
└── MambaTSR/                       ← External repo (auto-setup)
    ├── models/                     → Model architecture
    └── kernels/                    → CUDA kernels
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **CUDA 12.4+** (for GPU training)
- **Git**
- **16GB+ GPU memory** (recommended)

### Setup Instructions

#### 1. Clone Repository

```bash
git clone https://github.com/quoclam-doit/Plant_Disease.git
cd Plant_Disease
```

#### 2. Setup MambaTSR Dependencies

**Option A: Automatic Setup (Recommended)**

```bash
# Run setup script
python setup_mambatsr.py
```

This script will:

- Clone MambaTSR repository
- Install PyTorch nightly (CUDA 12.4)
- Compile selective_scan CUDA kernels
- Verify installation

**Option B: Manual Setup**

```bash
# 1. Clone MambaTSR
git clone https://github.com/VIDAR-Vision/MambaTSR.git

# 2. Install PyTorch nightly
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

# 3. Install dependencies
pip install timm matplotlib tqdm pillow

# 4. Compile CUDA kernels
cd MambaTSR/kernels/selective_scan

# Edit setup.py - Add this line to nvcc args:
# '-gencode', 'arch=compute_90,code=sm_90'

python setup.py install
cd ../../..
```

#### 3. Prepare Dataset

Download PlantVillage dataset and place in `Data/PlantVillage/PlantVillage-Dataset-master/`

Structure:

```
Data/
└── PlantVillage/
    └── PlantVillage-Dataset-master/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        ├── ... (39 classes)
```

#### 4. Train Model

```bash
python train_mambatsr_plantvillage.py
```

## 📊 Training Results

### Performance Metrics

| Metric                   | Value   |
| ------------------------ | ------- |
| **Best Val Accuracy**    | 98.96%  |
| **Final Train Accuracy** | 99.92%  |
| **Overfitting Gap**      | 0.96%   |
| **Training Time**        | 3:00:57 |
| **Epochs**               | 50      |
| **Batch Size**           | 32      |
| **Image Size**           | 64×64   |

### Model Architecture

- **Name**: VSSM-Tiny (Vision State Space Model)
- **Parameters**: 77,108,102 (77M)
- **Architecture**: `depths=[2,2,9,2]`, `dims=[96,192,384,768]`
- **Patch Size**: 4×4
- **Drop Path Rate**: 0.1

## 🔧 Technical Details

### Key Innovations

1. **CUDA Forward Compatibility**

   - RTX 5060 Ti has `sm_120` (compute capability 12.0)
   - CUDA doesn't support sm_120 compilation yet
   - Solution: Compile for `compute_90` → runs on `sm_120` via forward compatibility

2. **Image Size Optimization**

   - Standard: 224×224 → ~17 hours/epoch
   - Optimized: 64×64 → ~3.5 minutes/epoch
   - **16× speedup** with only ~1% accuracy trade-off

3. **PyTorch Nightly**
   - Required for new GPU support (sm_120)
   - CUDA 12.4+ compatibility
   - Latest kernels and optimizations

### System Requirements

**Minimum:**

- GPU: NVIDIA RTX 3060+ (8GB VRAM)
- RAM: 16GB
- Storage: 50GB

**Recommended:**

- GPU: NVIDIA RTX 4060 Ti / RTX 5060 Ti (16GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

### Environment

- **OS**: Windows 11 or Ubuntu 22.04+ (WSL2 supported)
- **Python**: 3.11+
- **CUDA**: 12.4+
- **PyTorch**: Nightly build (2.10.dev+cu124)

## 📚 Documentation

Comprehensive documentation available in `docs/`:

- **[Setup Guide](docs/guides/MAMBATSR_SETUP_GUIDE.md)** - Detailed setup instructions
- **[Training Guide](docs/guides/TRAINING_GUIDE.md)** - Training best practices
- **[How I Did It](docs/guides/HOW_I_DID_IT.md)** - Technical deep dive
- **[Training Results](docs/reports/TRAINING_RESULTS_REPORT.md)** - Complete results analysis
- **[Vision Mamba 2 Research](docs/reports/BAO_CAO_VISION_MAMBA_2.md)** - Future improvements

## 🎓 Research & References

### MambaTSR

This project uses **MambaTSR (Vision State Space Model)** for plant disease classification.

- **Repository**: https://github.com/VIDAR-Vision/MambaTSR
- **Paper**: "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model"
- **Authors**: VIDAR Vision Lab

**Key Features:**

- State-of-the-art vision architecture
- Linear complexity O(N) vs Transformer O(N²)
- Selective scan mechanism for efficient processing
- CUDA kernels for GPU acceleration

### PlantVillage Dataset

- **Source**: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Classes**: 39 plant disease categories
- **Images**: 54,304 color images (after filtering)
- **Resolution**: Various (resized to 64×64 for training)

## 🛠️ Development

### ⚠️ Important: Code Relationship

**This repository contains:**

- ✅ `train_mambatsr_plantvillage.py` - **NEW training script** adapted for PlantVillage dataset
- ✅ Custom data loading, configuration, and training pipeline
- ✅ Documentation and setup utilities

**This repository DOES NOT contain:**

- ❌ Model architecture (VSSM class) - imported from external MambaTSR repo
- ❌ CUDA kernels - compiled from MambaTSR repo during setup

**Relationship:**

| File                             | Location                                                                  | Purpose               | Author              |
| -------------------------------- | ------------------------------------------------------------------------- | --------------------- | ------------------- |
| `train_mambatsr_plantvillage.py` | **This repo**                                                             | PlantVillage training | ✅ **Our work**     |
| `MambaTSR/models/vmamba.py`      | [MambaTSR repo](https://github.com/1024AILab/MambaTSR)                    | VSSM model class      | ❌ Original authors |
| `MambaTSR/train.py`              | [MambaTSR repo](https://github.com/1024AILab/MambaTSR/blob/main/train.py) | ImageNet training     | ❌ Original authors |

**Key Differences:**

```python
# MambaTSR/train.py (Original - ImageNet)
- Dataset: ImageNet-1K (1.28M images, 1000 classes)
- Image size: 224×224
- Training: Distributed training on multiple GPUs
- Config: Command-line arguments

# train_mambatsr_plantvillage.py (Ours - PlantVillage)
- Dataset: PlantVillage (54K images, 39 classes)  ✅
- Image size: 64×64 (optimized for speed)        ✅
- Training: Single GPU (RTX 5060 Ti)             ✅
- Config: Python class (MambaTSRConfig)          ✅
- Features: Plotting, checkpointing, validation  ✅
```

**We wrote the training logic, but use the model architecture from the original MambaTSR repository.**

### Project Structure

```python
# Main training script (OUR CODE)
train_mambatsr_plantvillage.py
    ├── MambaTSRConfig          # Configuration class
    ├── prepare_dataset()       # Data loading for PlantVillage
    ├── build_model()           # Model construction (imports VSSM)
    ├── train_one_epoch()       # Training loop
    ├── validate()              # Validation
    └── save_checkpoint()       # Model saving

# Dependencies (EXTERNAL - auto-setup via setup_mambatsr.py)
MambaTSR/
    ├── models/vmamba.py        # VSSM class (imported)
    ├── models/VSSBlock.py      # Vision State Space blocks
    └── kernels/selective_scan/ # CUDA kernels
```

### Testing

```bash
# Run all tests
cd scripts/tests
python test_mambatsr_model.py
python test_selective_scan.py
python test_train_pipeline.py
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/Plant_Disease_MambaTSR.ipynb
```

## ❓ FAQ (Frequently Asked Questions)

### Q1: Tại sao code của bạn khác với MambaTSR gốc?

**A:** Chúng tôi **KHÔNG copy** code MambaTSR. Chúng tôi chỉ:

- ✅ **Import** class VSSM từ MambaTSR repo (như import library)
- ✅ **Viết mới** training script cho PlantVillage dataset
- ✅ **Tùy chỉnh** data loading, configuration, training loop

**Tương tự như:**

```python
# Bạn không viết lại PyTorch, chỉ import:
import torch
from torchvision import models

# Tương tự, chúng tôi import VSSM:
from MambaTSR.models.vmamba import VSSM
```

### Q2: File `train.py` ở đâu?

**A:** Có 2 file `train.py` khác nhau:

1. **`MambaTSR/train.py`** (Original)

   - Link: https://github.com/1024AILab/MambaTSR/blob/main/train.py
   - Purpose: Train VSSM on ImageNet-1K

2. **`train_mambatsr_plantvillage.py`** (Ours)
   - Link: https://github.com/quoclam-doit/Plant_Disease/blob/main/train_mambatsr_plantvillage.py
   - Purpose: Train VSSM on PlantVillage
   - **Đây là file chính của project**

### Q3: Làm sao chạy được mà không có model code?

**A:** Setup script (`setup_mambatsr.py`) sẽ:

1. Clone MambaTSR repository → Có `models/vmamba.py`
2. Compile CUDA kernels → Có `selective_scan`
3. Import vào `train_mambatsr_plantvillage.py` → Chạy được!

```bash
python setup_mambatsr.py  # Auto-download everything
python train_mambatsr_plantvillage.py  # Now it works!
```

### Q4: Có vi phạm license không?

**A:** KHÔNG! Chúng tôi:

- ✅ Credit original authors (see Citation section)
- ✅ Link to original repository
- ✅ Use their code as a library (not copy)
- ✅ Follow open-source best practices

**Giống như sử dụng PyTorch, TensorFlow - hoàn toàn hợp lệ!**

### Q5: Tại sao không push MambaTSR/ lên GitHub?

**A:** Vì:

- ❌ Đó là code của người khác
- ❌ CUDA binaries rất lớn (~200MB)
- ❌ Không cần thiết (users có thể auto-download)
- ✅ Setup script handle việc này

**Best practice: Link to original, don't copy!**

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{plant_disease_mambatsr_2025,
  author = {quoclam-doit},
  title = {Plant Disease Classification with MambaTSR},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/quoclam-doit/Plant_Disease}
}

@article{mambatsr2024,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={VIDAR Vision Lab},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📧 Contact

- **Author**: quoclam-doit
- **GitHub**: [@quoclam-doit](https://github.com/quoclam-doit)
- **Repository**: [Plant_Disease](https://github.com/quoclam-doit/Plant_Disease)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MambaTSR is from [VIDAR-Vision/MambaTSR](https://github.com/VIDAR-Vision/MambaTSR) and may have its own license.

## 🙏 Acknowledgments

- **MambaTSR Team** - For the amazing Vision Mamba architecture
- **PlantVillage** - For the comprehensive plant disease dataset
- **PyTorch Team** - For nightly builds supporting new GPUs
- **NVIDIA** - For CUDA forward compatibility

---

⭐ **Star this repository if you find it helpful!**

💡 **Check out the [docs/](docs/) folder for detailed guides and reports.**

🚀 **Happy Training!**
