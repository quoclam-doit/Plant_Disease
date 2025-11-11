# GIẢI PHÁP: SỬ DỤNG PYTORCH NIGHTLY VỚI sm_120 TRÊN WINDOWS

## 🎯 Dựa trên thông tin từ PyTorch Forum

**Link:** https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099/2

**Tóm tắt:** PyTorch nightly đã hỗ trợ sm_120 (Blackwell), nhưng chỉ có cho Linux.

---

## ✅ GIẢI PHÁP: SỬ DỤNG WSL2

### 1. Cài đặt WSL2 (Windows Subsystem for Linux)

**Bước 1: Enable WSL (PowerShell Administrator)**

```powershell
wsl --install
# Hoặc nếu đã có WSL1:
wsl --set-default-version 2
```

**Bước 2: Install Ubuntu**

```powershell
wsl --install -d Ubuntu-22.04
```

**Bước 3: Restart máy**

---

### 2. Setup CUDA trong WSL2

**WSL2 tự động access CUDA driver từ Windows!**

```bash
# Trong WSL2 Ubuntu terminal:
# Kiểm tra GPU
nvidia-smi

# Output mong đợi:
# NVIDIA GeForce RTX 5060 Ti
# Driver Version: 13.0
```

**⚠️ QUAN TRỌNG:**

- KHÔNG cần cài CUDA Toolkit trong WSL2
- Windows CUDA driver đã shared vào WSL2
- Chỉ cần install PyTorch nightly

---

### 3. Setup Python Environment trong WSL2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip -y

# Create virtual environment
cd /mnt/g/Dataset  # Access Windows drive
python3.11 -m venv .venv_wsl

# Activate
source .venv_wsl/bin/activate
```

---

### 4. Install PyTorch Nightly với sm_120 Support

```bash
# Activate venv
source .venv_wsl/bin/activate

# Install PyTorch nightly với CUDA 12.8 (có sm_120)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify
python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
```

**Expected output:**

```
2.7.0.dev20250131+cu128
['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
                                                                              ↑ ↑
                                                                        CÓ sm_120!
```

---

### 5. Test CUDA Operations

```bash
python -c "import torch; x = torch.randn(1, 3, 32, 32).cuda(); import torch.nn.functional as F; y = F.max_pool2d(x, 2); print(f'✓ Success! Output: {y.shape}')"
```

**Expected:**

```
✓ Success! Output: torch.Size([1, 3, 16, 16])
```

---

### 6. Install Dependencies

```bash
# Install các dependencies cần thiết
pip install timm==0.4.12 einops fvcore tensorboard
pip install numpy pandas matplotlib seaborn scikit-learn

# torchvision vẫn đang WIP, có thể dùng version cũ tạm:
pip install torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
# Hoặc đợi torchvision nightly support sm_120
```

---

### 7. Compile selective_scan trong WSL2

```bash
cd /mnt/g/Dataset/MambaTSR/kernels/selective_scan

# Set environment
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

# Build
pip install --no-build-isolation -e .
```

**⚠️ Lưu ý:**

- Trong WSL2 (Linux), KHÔNG CẦN fix M_LOG2E và BOOL_SWITCH
- Code sẽ compile sạch sẽ với GCC

---

### 8. Run Training

```bash
cd /mnt/g/Dataset

# Copy notebook hoặc Python script
# Run training
python MambaTSR/train.py --config configs/vssm1/vssm_tiny_224.yaml
```

---

## 📊 SO SÁNH GIẢI PHÁP

| Giải pháp                           | Thời gian setup | Độ khó           | Success rate | Performance       |
| ----------------------------------- | --------------- | ---------------- | ------------ | ----------------- |
| **WSL2 + PyTorch nightly**          | 30-60 phút      | ⭐⭐ Trung bình  | 90%          | 100% (native GPU) |
| Build PyTorch from source (Windows) | 4-6 giờ         | ⭐⭐⭐⭐ Rất khó | 60%          | 100%              |
| Build PyTorch from source (WSL2)    | 3-4 giờ         | ⭐⭐⭐ Khó       | 75%          | 100%              |
| Google Colab                        | 15 phút         | ⭐ Dễ            | 100%         | 85% (T4 GPU)      |
| Cloud GPU                           | 30 phút         | ⭐⭐ Dễ          | 100%         | 100% ($$$)        |

---

## ✅ KHUYẾN NGHỊ

### **Phương án TỐI ƯU: WSL2 + PyTorch Nightly**

**Ưu điểm:**

- ✅ Nhanh (30-60 phút setup)
- ✅ Dùng đúng RTX 5060 Ti của bạn
- ✅ PyTorch official build (không tự compile)
- ✅ Performance đầy đủ
- ✅ Free, không mất tiền

**Nhược điểm:**

- ⚠️ Cần học WSL2 cơ bản (không khó)
- ⚠️ torchvision có thể chưa có nightly (dùng tạm version cũ)

---

## 🚀 TIMELINE DỰ KIẾN

```
[15 phút] Install WSL2 + Ubuntu
[10 phút] Setup Python environment
[10 phút] Install PyTorch nightly
[5 phút]  Verify CUDA works
[15 phút] Install dependencies
[20 phút] Compile selective_scan
────────────────────────────────
TOTAL: ~75 phút (1.5 giờ)

→ Sau đó có thể train MambaTSR ngay!
```

---

## 📝 CHECKLIST

**Trước khi bắt đầu:**

- [ ] Windows 11 version 21H2 trở lên
- [ ] NVIDIA Driver 13.0 (đã có ✓)
- [ ] ~10GB disk space trống
- [ ] Internet connection

**Các bước thực hiện:**

- [ ] Install WSL2
- [ ] Install Ubuntu 22.04
- [ ] Verify nvidia-smi trong WSL2
- [ ] Create Python venv
- [ ] Install PyTorch nightly cu128
- [ ] Test CUDA operations
- [ ] Install dependencies
- [ ] Compile selective_scan
- [ ] Test MambaTSR forward pass
- [ ] Start training

---

## ⚠️ TROUBLESHOOTING

### Issue: nvidia-smi không work trong WSL2

**Solution:**

```bash
# Check Windows GPU driver version
# Trong Windows PowerShell:
nvidia-smi

# Phải có Driver Version: 13.0+
# Nếu không, update NVIDIA driver
```

### Issue: PyTorch không thấy GPU

**Solution:**

```bash
# Kiểm tra CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Nếu False, check WSL2 kernel version:
wsl --version

# Cần WSL2 kernel 5.10.16+
# Update: wsl --update
```

### Issue: torchvision không compatible

**Solution:**

```bash
# Option 1: Dùng version cũ tạm
pip install torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Option 2: Build torchvision from source (thêm 30 phút)
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install
```

---

## 🎯 KẾT LUẬN

**WSL2 + PyTorch Nightly = GIẢI PHÁP TỐI ưu**

- Nhanh, đơn giản, official support
- Tận dụng đầy đủ RTX 5060 Ti
- Không mất tiền
- Có thể bắt đầu ngay hôm nay!

**→ Đề xuất: Thử phương án này trước, nếu fail mới xét các phương án khác**

---

**Nguồn tham khảo:**

- PyTorch Forum: https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099/2
- WSL2 CUDA Guide: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- PyTorch Nightly: https://pytorch.org/get-started/locally/#start-locally

**Ngày tạo:** 10/11/2025  
**Dựa trên:** Thông tin từ ptrblck (PyTorch core developer)
