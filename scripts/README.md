# 🔧 Scripts

## 📁 Folder Structure

```
scripts/
├── tests/  → Test scripts và verification
└── shell/  → Shell scripts để chạy training
```

## 🧪 Test Scripts (11 files)

### Environment Checks

- `check_gpu_compute.py` - Kiểm tra GPU compute capability
- `check_mambatsr_env.py` - Kiểm tra MambaTSR environment
- `check_versions.py` - Kiểm tra versions của libraries

### CUDA Tests

- `test_cuda.py` - Test CUDA functionality
- `test_selective_scan.py` - Test selective_scan kernel

### Import Tests

- `test_direct_import.py` - Test direct imports
- `test_import_simple.py` - Test simple imports
- `test_step_imports.py` - Test step-by-step imports

### Model Tests

- `test_mambatsr_model.py` - Test MambaTSR model
- `test_train_pipeline.py` - Test training pipeline

### Verification

- `verify_selective_scan.py` - Verify selective_scan compilation

## 🐚 Shell Scripts (3 files)

- `run_training.sh` - WSL training script
- `start_training.sh` - Linux training script
- `START_TRAINING.bat` - Windows batch script

## 💡 Usage

### Run Tests

```bash
# Check environment
python scripts/tests/check_gpu_compute.py
python scripts/tests/check_versions.py

# Test CUDA
python scripts/tests/test_cuda.py

# Test model
python scripts/tests/test_mambatsr_model.py
```

### Run Training (Shell)

```bash
# WSL2
./scripts/shell/run_training.sh

# Windows
scripts\shell\START_TRAINING.bat
```

## 📌 Note

Main training script is at: `../train_mambatsr_plantvillage.py`
