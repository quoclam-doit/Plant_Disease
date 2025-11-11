# 📚 Documentation

## 📁 Folder Structure

```
docs/
├── reports/        → Báo cáo kỹ thuật và phân tích
├── guides/         → Hướng dẫn setup và troubleshooting
└── presentations/  → Tài liệu trình bày cho giảng viên
```

## 📋 File Summary

### 📊 Reports (4 files)

- `BAO_CAO_VISION_MAMBA_2.md` - Nghiên cứu Vision Mamba 2 (VSSD)
- `BAO_CAO_VAN_DE_KY_THUAT.md` - Báo cáo vấn đề kỹ thuật
- `TRAINING_RESULTS_REPORT.md` - Kết quả training chi tiết
- `CODE_SUMMARY_FOR_TEACHER.md` - Tóm tắt code cho giảng viên

### 📖 Guides (7 files)

- `HOW_I_DID_IT.md` - Technical deep dive (10 sections)
- `MAMBATSR_SETUP_GUIDE.md` - Setup instructions
- `TRAINING_GUIDE.md` - Training guide
- `QUICK_START.md` - Quick start guide
- `SOLUTION_WSL2_PYTORCH_NIGHTLY.md` - WSL2 + PyTorch nightly solution
- `DANG_CHUNG_PYTORCH_CUDA_COMPATIBILITY.md` - CUDA compatibility guide
- `MAMBATSR_RTX5060TI_FINAL_STATUS.md` - Final status report

### 🎤 Presentations (4 files)

- `PRESENTATION_FOR_TEACHER.md` - 16-slide presentation
- `QUICK_REFERENCE_FOR_PRESENTATION.md` - Q&A cheat sheet
- `FINAL_SUMMARY.md` - Final summary
- `README_MAMBATSR_STATUS.md` - MambaTSR status

## 🎯 Key Results

**Training Achievement:**

- Model: MambaTSR (VSSM-Tiny) - 77M parameters
- Dataset: PlantVillage - 54,304 images, 39 classes
- **Best Accuracy: 98.96%** (Epoch 48)
- Training Time: 3 hours 0 minutes 57 seconds
- Hardware: RTX 5060 Ti 16GB (sm_120)

**Technical Solutions:**

1. WSL2 + PyTorch nightly (support sm_120)
2. CUDA forward compatibility (compute_90 → sm_120)
3. Image size optimization (64×64 = 16× speedup)

## 📌 Related Files

Main training script: `../train_mambatsr_plantvillage.py`  
Plotting script: `../generate_training_plots.py`  
Test scripts: `../scripts/tests/`  
Notebooks: `../notebooks/`
