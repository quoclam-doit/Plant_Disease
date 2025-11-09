# MambaTSR for PlantVillage - Setup Status

**Date:** November 9, 2025  
**Status:** 🟡 Ready to Install Build Tools

---

## 📊 Current Environment Status

### ✅ Completed Setup (90%)

| Component | Status | Version | Notes |
|-----------|--------|---------|-------|
| Python | ✅ | 3.11.9 | In virtual environment |
| PyTorch | ✅ | 2.6.0+cu124 | Matches CUDA 12.4 |
| CUDA | ✅ | 12.4 | System CUDA |
| GPU | ⚠️ | RTX 5060 Ti (16GB) | See warning below |
| NumPy | ✅ | 2.2.6 | |
| TorchVision | ✅ | 0.21.0+cu124 | |
| timm | ✅ | 0.4.12 | PyTorch Image Models |
| einops | ✅ | 0.8.1 | Tensor operations |
| fvcore | ✅ | 0.1.5 | Facebook core library |
| Matplotlib | ✅ | Installed | Visualization |
| Seaborn | ✅ | Installed | Statistical plots |
| scikit-learn | ✅ | Installed | Metrics |

### ⏳ Pending Installation (10%)

| Component | Status | Required For | Action |
|-----------|--------|--------------|--------|
| **Visual Studio Build Tools** | ❌ | Compile CUDA kernels | [Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| **selective_scan_cuda_core** | ❌ | MambaTSR SS2D operation | Install after Build Tools |

---

## ⚠️ Important Warning: RTX 5060 Ti

```
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible 
with the current PyTorch installation.
```

**Impact:**
- PyTorch 2.6.0 does NOT support compute capability 12.0 (RTX 50 series)
- Model may run slower or not utilize full GPU potential
- May fall back to compatibility mode

**Solutions:**
1. **Option A - Upgrade to PyTorch Nightly** (Recommended):
   ```bash
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
   ```
   - Pros: Full RTX 5060 Ti support
   - Cons: Nightly builds may be unstable

2. **Option B - Wait for PyTorch 2.7+**:
   - Expected Q1 2026
   - Stable release with RTX 50 series support

3. **Option C - Continue with current setup**:
   - May work in compatibility mode
   - Performance not optimal

---

## 📁 Project Files

### Created Files

```
G:\Dataset\
├── Plant_Disease_MambaTSR.ipynb       ✅ Main training notebook
├── check_mambatsr_env.py              ✅ Environment verification script
├── MAMBATSR_SETUP_GUIDE.md            ✅ Detailed setup instructions
├── QUICK_START.md                     ✅ Quick reference guide
├── README_MAMBATSR_STATUS.md          ✅ This file
└── MambaTSR/                          ✅ Cloned repository
    ├── models/
    │   ├── VSSBlock_utils.py          ✅ Super_Mamba class (line 59)
    │   ├── VSSBlock.py                ✅ VSSBlock implementation
    │   ├── ConvNet.py                 ✅ Embedding network
    │   └── vmamba.py                  ✅ SS2D, Mlp
    └── kernels/
        └── selective_scan/            ⏳ Needs compilation
            ├── setup.py               ✅
            └── csrc/                  ✅ CUDA source files
```

### Existing Files

```
G:\Dataset\
├── Plant_Disease_YOLOv4_Ensemble.ipynb  ✅ Previous ensemble notebook (working)
├── models/                              ✅ Trained ensemble models
│   ├── densenet121_best.pth
│   ├── efficientnet_b3_best.pth
│   ├── resnet50_best.pth
│   └── inception_v3_best.pth
└── Data/
    └── PlantVillage/                    ✅ Dataset (54,305 images, 39 classes)
```

---

## 🎯 Next Steps (In Order)

### Step 1: Install Build Tools (30-40 min)

1. Download: https://aka.ms/vs/17/release/vs_BuildTools.exe
2. Run installer
3. Select workload: **Desktop development with C++**
4. Wait for installation to complete
5. Restart computer (recommended)

### Step 2: Compile Selective Scan (5-10 min)

```powershell
cd G:\Dataset
.\.venv\Scripts\Activate.ps1
cd MambaTSR\kernels\selective_scan
pip install --no-build-isolation -e .
```

### Step 3: Verify Installation

```powershell
python check_mambatsr_env.py
```

Look for:
```
✅ Selective Scan Kernel: Installed
✓ Super_Mamba model created successfully
```

### Step 4: Run Training

1. Open `Plant_Disease_MambaTSR.ipynb` in VS Code
2. Select kernel: `.venv (Python 3.11)`
3. Run all cells
4. Training will take several hours

---

## 📖 Documentation Files

1. **QUICK_START.md** - Fast reference for common tasks
2. **MAMBATSR_SETUP_GUIDE.md** - Detailed setup instructions with troubleshooting
3. **check_mambatsr_env.py** - Automated environment checker
4. **README_MAMBATSR_STATUS.md** - This file (current status)

---

## 🔍 Verification Checklist

Before training:

- [x] Python 3.11 virtual environment
- [x] PyTorch 2.6.0+cu124 installed
- [x] CUDA 12.4 available
- [x] All Python dependencies (timm, einops, fvcore, etc.)
- [x] MambaTSR repository cloned
- [x] Notebook created and ready
- [ ] Visual Studio Build Tools installed
- [ ] selective_scan_cuda_core compiled
- [ ] Environment check passes all tests

---

## 🎓 Model Architecture (from VSSBlock_utils.py line 59)

```python
class Super_Mamba(nn.Module):
    """
    Architecture:
    1. ConvNet: RGB → Feature embedding (3 → 64 → 128 → 3 channels)
    2. 6x [PatchMerging2D + VSSBlock]:
       - PatchMerging: H,W → H/2,W/2; C → 2C
       - VSSBlock: Selective Scan 2D (SS2D) operation
    3. Classifier: LayerNorm → AvgPool → Linear(39 classes)
    
    Parameters: ~90k (extremely lightweight!)
    """
```

### Adaptations for PlantVillage:
- ✅ `num_classes=39` (instead of 43 traffic signs)
- ✅ Image size: 32x32 (as per MambaTSR paper)
- ✅ Data augmentation: ColorJitter, Flip, Rotation
- ✅ Normalization: ImageNet stats (0.485, 0.456, 0.406)

---

## 💡 Tips

1. **Build Tools installation is THE bottleneck**
   - Takes 30-40 minutes
   - Requires ~8GB download
   - Restart after installation recommended

2. **Selective scan compilation is SLOW**
   - Takes 5-10 minutes
   - No progress bar shown
   - Don't cancel! Wait patiently

3. **RTX 5060 Ti compatibility**
   - Consider PyTorch nightly for full GPU utilization
   - Or accept slower training with current setup

4. **Alternative: Run on Google Colab**
   - Free GPU (T4 or L4)
   - Pre-installed Build Tools
   - Faster to get started
   - But limited runtime (12 hours)

---

## 📊 Expected Results

Based on MambaTSR paper (traffic signs):
- **Accuracy:** 95-98% on test set
- **Parameters:** ~90k (very lightweight)
- **Training time:** 2-4 hours (depends on GPU)

For PlantVillage (39 classes, 54k images):
- **Expected accuracy:** 90-95% (more classes, similar architecture)
- **Training time:** 4-8 hours (larger dataset)

---

## 🆚 Comparison with Ensemble Models

| Model | Parameters | Accuracy (Expected) | Training Time |
|-------|------------|---------------------|---------------|
| EfficientNetB3 | 12M | 97%+ | 2-3 hours |
| ResNet50 | 25M | 96%+ | 2-3 hours |
| DenseNet121 | 8M | 96%+ | 2-3 hours |
| InceptionV3 | 24M | 96%+ | 3-4 hours |
| **Ensemble** | 69M | **98%+** | 8-12 hours |
| **Super_Mamba** | **90k** | **90-95%** | **4-8 hours** |

**MambaTSR Advantages:**
- ⚡ **700x fewer parameters** than ensemble
- 💾 **Much smaller model size** (~350KB vs 270MB)
- 🚀 **Faster inference** (good for deployment)
- 📱 **Mobile-friendly** (can run on edge devices)

**MambaTSR Trade-offs:**
- 📉 Slightly lower accuracy (90-95% vs 98%+)
- 🔧 More complex setup (CUDA kernels required)
- 🆕 Newer architecture (less mature than CNNs)

---

## 📞 Support & Resources

- **MambaTSR Paper:** "MambaTSR: You Only Need 90k Parameters for Traffic Sign Recognition"
- **GitHub:** https://github.com/1024AILab/MambaTSR
- **PyTorch Forums:** https://discuss.pytorch.org/
- **VS Code + Python:** https://code.visualstudio.com/docs/python/

---

**Status Updated:** November 9, 2025 23:15  
**Next Action:** Install Visual Studio Build Tools  
**ETA to Training:** 40-50 minutes (after Build Tools installation)
