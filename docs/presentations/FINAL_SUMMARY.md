# 📋 Tổng Kết: MambaTSR trên RTX 5060 Ti - HOÀN THÀNH ✅

**Ngày**: 10 Tháng 11, 2025  
**GPU**: NVIDIA GeForce RTX 5060 Ti 16GB (Compute Capability 12.0 / sm_120)  
**Status**: ✅ **SẴN SÀNG TRAIN!**

---

## 🎯 Mục tiêu

Train model **MambaTSR** trên dataset **PlantVillage** (54,304 ảnh, 38 disease classes) để phát hiện bệnh cây trồng.

---

## ✅ Những gì đã hoàn thành

### 1. Setup Environment ✅

- WSL2 Ubuntu 22.04
- CUDA Toolkit 12.4
- Python 3.11 virtual environment
- PyTorch 2.10.0.dev (nightly) với CUDA 12.8 support

### 2. Dependency Resolution ✅

**Vấn đề gặp phải:**

- torchvision 0.15.2+cu117 không tương thích với PyTorch 2.10
- timm 0.4.12 gây segmentation fault

**Giải quyết:**

- ✅ Update torchvision → 0.25.0.dev (nightly cu128)
- ✅ Update timm → 1.0.22
- ✅ Tất cả imports hoạt động ổn định

### 3. Selective Scan Compilation - CRITICAL! ✅

**Vấn đề lớn:** RTX 5060 Ti có sm_120 (Blackwell) nhưng CUDA 12.4 không support!

**Giải quyết bằng Forward Compatibility:**

- Compile với `compute_90,code=compute_90` thay vì `sm_90,code=sm_90`
- Tạo PTX intermediate code → GPU driver JIT-compile cho sm_120
- ✅ Hoạt động hoàn hảo!

**Compiled Extensions:**

- selective_scan_cuda_core: 33.9 MB
- selective_scan_cuda_ndstate: 32.8 MB
- selective_scan_cuda_oflex: 30.8 MB
- Total: ~97 MB

### 4. Model Testing ✅

**Test Results:**

```
✓ Model loads: 2,951,942 parameters
✓ GPU forward pass: SUCCESS!
  - Input: [2, 3, 224, 224]
  - Output: [2, 38]
  - Memory: 128.7 MB peak
  - Batch=2 chỉ dùng 116 MB!
```

### 5. Training Pipeline ✅

**Created:**

- `train_mambatsr_plantvillage.py` - Full training script
- `test_train_pipeline.py` - Pipeline validator
- `start_training.sh` - One-click start script

**Tested:**

- ✅ Dataset loading: 43,440 train / 10,860 val
- ✅ Model building: 3M parameters
- ✅ Forward pass: Working
- ✅ Training started: backward pass OK
- ✅ GPU utilized properly

---

## 📊 Training Configuration

### Model: VSSM-Tiny

```python
depths = [2, 2, 9, 2]
dims = [96, 192, 384, 768]
parameters = 2,951,942
```

### Dataset: PlantVillage

```
Total: 54,304 images
Classes: 38 plant diseases
Split: 80/20 train/val
Augmentation: flip, rotate, color jitter
```

### Hyperparameters

```python
batch_size = 32          # Optimal for RTX 5060 Ti
epochs = 50
learning_rate = 1e-4
optimizer = AdamW
scheduler = Cosine + Warmup
```

### Expected Performance

```
Time per epoch: ~10-15 minutes
Total time (50 epochs): ~8-12 hours
Speed: ~10-15 it/s (batch=32)
GPU memory: 6-8 GB peak
```

---

## 🔧 Technical Solutions

### Problem 1: sm_120 không được CUDA 12.4 support

**Solution:** Forward compatibility

```python
# Before (failed):
cc_flag.extend(["-gencode", "arch=compute_90,code=sm_90"])

# After (works!):
cc_flag.extend(["-gencode", "arch=compute_90,code=compute_90"])
```

### Problem 2: Loaded old .so files thay vì new ones

**Solution:** Remove source directory .so files

```bash
cd MambaTSR/kernels/selective_scan
rm *.so  # Force Python to load from site-packages
```

### Problem 3: Dependency incompatibilities

**Solution:** Update to compatible versions

```
torchvision: 0.15.2 → 0.25.0.dev (nightly)
timm: 0.4.12 → 1.0.22
```

---

## 📁 Project Structure

```
G:\Dataset\
├── train_mambatsr_plantvillage.py    # Main training script
├── test_train_pipeline.py            # Pipeline tester
├── start_training.sh                 # Quick start
├── TRAINING_GUIDE.md                 # User guide (Vietnamese)
├── MAMBATSR_RTX5060TI_FINAL_STATUS.md # Technical details
│
├── Data/PlantVillage/
│   └── PlantVillage-Dataset-master/  # 54,304 images, 38 classes
│
├── MambaTSR/
│   ├── models/vmamba.py              # MambaTSR model
│   └── kernels/selective_scan/       # Compiled CUDA kernels
│       ├── selective_scan_cuda_core.so (33.9 MB)
│       ├── selective_scan_cuda_ndstate.so (32.8 MB)
│       └── selective_scan_cuda_oflex.so (30.8 MB)
│
└── models/MambaTSR/                  # Output directory (will be created)
    ├── mambatsr_best.pth             # Best checkpoint
    ├── mambatsr_epoch_X.pth          # Periodic checkpoints
    ├── training_history.json         # Metrics
    └── class_names.json              # Class mapping
```

---

## 🚀 How to Start Training

### Quick Start (Recommended):

```bash
wsl bash /mnt/g/Dataset/start_training.sh
```

### Manual Start:

```bash
wsl bash -c "/mnt/g/Dataset/.venv_wsl/bin/python /mnt/g/Dataset/train_mambatsr_plantvillage.py"
```

### Test First (Safer):

```bash
wsl bash -c "/mnt/g/Dataset/.venv_wsl/bin/python /mnt/g/Dataset/test_train_pipeline.py"
```

---

## 📈 Expected Results

### After 50 epochs, expect:

- **Validation Accuracy**: 85-95% (PlantVillage is relatively clean dataset)
- **Training Time**: 8-12 hours
- **Best Model**: Saved as `models/MambaTSR/mambatsr_best.pth`
- **Checkpoints**: Every 5 epochs

### Monitoring:

Terminal will show real-time:

- Loss curves (train & val)
- Accuracy (train & val)
- Learning rate schedule
- GPU memory usage
- Training speed (it/s)
- Best model updates

---

## 💾 GPU Memory Budget

**RTX 5060 Ti 16GB:**

```
Model loading:     ~20 MB
Batch (32 images): ~2-4 GB
Forward pass:      ~1-2 GB
Backward pass:     ~2-3 GB
Optimizer state:   ~1-2 GB
─────────────────────────
Peak usage:        ~6-8 GB
Available:         ~8-10 GB ✅ Plenty of headroom!
```

**If Out of Memory:**

```python
batch_size = 16  # Reduce by half
# or
batch_size = 8   # Reduce to 1/4
```

---

## 🎓 Key Learnings

### 1. Forward Compatibility is Powerful

Compile with `compute_X,code=compute_X` để GPU mới hơn có thể chạy code cũ hơn.

### 2. CUDA Toolkit Limitations

CUDA 12.4 chưa support sm_120 native → Phải dùng forward compatibility.

### 3. Dependency Hell is Real

PyTorch nightly cần torchvision nightly. timm 0.4.12 quá cũ cho PyTorch 2.10.

### 4. Python Import Order Matters

Source directory .so files được load trước site-packages → Phải remove old files.

### 5. RTX 5060 Ti is Capable

16GB VRAM đủ để train models cỡ nhỏ-trung bình một cách thoải mái.

---

## 📚 Documentation Files

1. **MAMBATSR_RTX5060TI_FINAL_STATUS.md** - Kỹ thuật chi tiết
2. **TRAINING_GUIDE.md** - Hướng dẫn train (Vietnamese)
3. **README_MAMBATSR_STATUS.md** - Status cũ (deprecated)
4. **BAO_CAO_VAN_DE_KY_THUAT.md** - Technical report (Vietnamese)

---

## ✅ Final Checklist

- [x] WSL2 + Ubuntu 22.04 setup
- [x] CUDA 12.4 installed
- [x] PyTorch nightly (2.10.0.dev+cu128)
- [x] All dependencies resolved
- [x] selective_scan compiled (sm_90 → sm_120 forward compat)
- [x] MambaTSR model imports
- [x] GPU forward pass works
- [x] Dataset loaded (54,304 images)
- [x] Training pipeline tested
- [x] Scripts created
- [x] Documentation complete
- [ ] **→ READY TO TRAIN! 🚀**

---

## 🎉 Success Criteria Met

✅ **Model loads** - 3M parameters  
✅ **GPU works** - RTX 5060 Ti (sm_120) recognized  
✅ **Forward pass** - 116 MB for batch=2  
✅ **Backward pass** - Tested in training loop  
✅ **Dataset ready** - 54,304 images loaded  
✅ **Scripts ready** - Training pipeline complete  
✅ **Documentation** - Full guides available

---

## 🚀 Next Action

**BẮT ĐẦU TRAIN:**

```bash
wsl bash /mnt/g/Dataset/start_training.sh
```

Có thể để chạy qua đêm. Kết quả sẽ lưu trong `models/MambaTSR/`.

---

## 📞 Support & Troubleshooting

**If issues occur:**

1. **Check GPU**: `wsl nvidia-smi`
2. **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check imports**: `python test_direct_import.py`
4. **Check pipeline**: `python test_train_pipeline.py`

**Common issues:**

- Out of memory → Reduce batch_size
- Slow training → Check GPU utilization
- Import errors → Re-run setup scripts

---

## 🏆 Achievement Unlocked!

✅ **MambaTSR hoạt động trên RTX 5060 Ti (sm_120)!**  
✅ **PlantVillage dataset sẵn sàng!**  
✅ **Training pipeline hoàn chỉnh!**  
✅ **Documentation đầy đủ!**

**Status**: **100% READY FOR PRODUCTION TRAINING** 🎉

---

**Tóm lại: Đã setup xong hoàn toàn, sẵn sàng train ngay!** 🚀🌱

Chỉ cần chạy:

```bash
wsl bash /mnt/g/Dataset/start_training.sh
```

**Chúc may mắn với training! 🍀**

---

_Report compiled: November 10, 2025_  
_By: GitHub Copilot_  
_For: PlantVillage Disease Detection with MambaTSR_
