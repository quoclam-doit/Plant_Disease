# MambaTSR Setup Guide - Fix CUDA Environment

## 🎯 Tình trạng hiện tại

✅ **Đã hoàn thành:**

- PyTorch 2.6.0 với CUDA 12.4 (match với system CUDA)
- Virtual environment đã setup đúng
- Notebook MambaTSR đã tạo hoàn chỉnh

❌ **Còn thiếu:**

- **Microsoft Visual C++ Build Tools** (để compile CUDA kernels)

---

## 📦 Bước 1: Cài đặt Microsoft Visual C++ Build Tools

### Option A: Cài đặt đầy đủ (Recommended)

1. **Download Visual Studio Build Tools:**

   - Link: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Hoặc: https://aka.ms/vs/17/release/vs_BuildTools.exe

2. **Cài đặt với workloads sau:**

   - ✅ **Desktop development with C++**
   - ✅ **MSVC v143 - VS 2022 C++ x64/x86 build tools** (Latest)
   - ✅ **Windows 10 SDK** (10.0.20348.0 or latest)
   - ✅ **CMake tools for Windows**

3. **Dung lượng cần:** ~7-8 GB

### Option B: Minimal Install (nhanh hơn)

Chỉ cài đặt:

- MSVC compiler
- Windows SDK
- CMake

---

## 🔧 Bước 2: Sau khi cài Build Tools

### 1. Verify compiler

Mở **Developer Command Prompt for VS 2022** và chạy:

```cmd
cl
```

Nếu thấy "Microsoft (R) C/C++ Optimizing Compiler" là thành công.

### 2. Cài đặt selective_scan kernel

```powershell
# Activate virtual environment
cd G:\Dataset
.\.venv\Scripts\Activate.ps1

# Install selective_scan
cd G:\Dataset\MambaTSR\kernels\selective_scan
pip install --no-build-isolation -e .
```

### 3. Verify installation

```python
python -c "import selective_scan_cuda_core; print('✓ Selective scan installed successfully!')"
```

---

## 🚀 Bước 3: Chạy MambaTSR Notebook

1. Open: `G:\Dataset\Plant_Disease_MambaTSR.ipynb`
2. Select kernel: `.venv` (Python 3.11)
3. Run all cells

---

## ⚠️ Troubleshooting

### Issue 1: "cl.exe not found"

**Solution:**

- Cài đặt lại Visual Studio Build Tools
- Hoặc add to PATH: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64`

### Issue 2: "ninja not found warning"

**Solution:**

```powershell
pip install ninja
```

### Issue 3: CUDA version mismatch

**Current status:** ✅ Fixed (PyTorch 2.6.0+cu124 matches CUDA 12.4)

### Issue 4: Compilation quá lâu

- Selective scan kernel có thể mất 5-10 phút để compile lần đầu
- Progress không hiển thị - hãy kiên nhẫn đợi
- Nếu quá 20 phút, hãy cancel và thử lại

---

## 🔍 Alternative: Sử dụng Pre-compiled Wheels

Nếu không muốn cài Build Tools, có thể thử:

1. **Tìm pre-compiled wheel** (nếu có):

   ```powershell
   pip install selective-scan-cuda --find-links https://github.com/...
   ```

2. **Hoặc sử dụng WSL2** (Linux environment):

   - Easier to compile CUDA code
   - Không cần Visual Studio

3. **Hoặc sử dụng Docker container** với pre-built environment

---

## 📊 Expected Timeline

| Task                   | Duration       | Status     |
| ---------------------- | -------------- | ---------- |
| Download Build Tools   | 5-10 min       | ⏳ Pending |
| Install Build Tools    | 10-20 min      | ⏳ Pending |
| Compile selective_scan | 5-10 min       | ⏳ Pending |
| **Total**              | **~30-40 min** | ⏳         |

---

## ✅ Verification Checklist

Sau khi setup xong, verify:

```python
# 1. Check PyTorch
import torch
print(f"PyTorch: {torch.__version__}")  # Should be 2.6.0+cu124
print(f"CUDA: {torch.cuda.is_available()}")  # Should be True

# 2. Check selective_scan
import selective_scan_cuda_core
print("✓ Selective scan imported successfully")

# 3. Check MambaTSR components
from models.ConvNet import ConvNet
from models.VSSBlock import VSSBlock
from models.vmamba import SS2D
print("✓ All MambaTSR components imported")

# 4. Test model creation
from models.VSSBlock_utils import Super_Mamba
model = Super_Mamba(dims=3, depth=6, num_classes=39)
print(f"✓ Super_Mamba model created: {sum(p.numel() for p in model.parameters()):,} parameters")
```

---

## 📞 Support

Nếu gặp vấn đề:

1. Check error message carefully
2. Google "pytorch cuda extension windows <your error>"
3. Check PyTorch forums: https://discuss.pytorch.org/

---

## 🎓 Lưu ý cho thầy

Notebook `Plant_Disease_MambaTSR.ipynb` đã được tạo theo đúng:

✅ **Architecture từ VSSBlock_utils.py line 59**: Class `Super_Mamba`
✅ **Tuân thủ cấu trúc MambaTSR**: ConvNet → PatchMerging → VSSBlock
✅ **Adapted cho PlantVillage**: `num_classes=39` (thay vì 43 traffic signs)
✅ **Complete training pipeline**: DataLoader, Optimizer, Scheduler, Metrics
✅ **Production ready**: Checkpoints, Early stopping, Visualization

Chỉ cần cài Build Tools là có thể chạy được!

---

**Last updated:** November 9, 2025
**Status:** ⏳ Waiting for Build Tools installation
