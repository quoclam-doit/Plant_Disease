"""
MambaTSR Environment Check Script
Kiểm tra tất cả dependencies cần thiết để chạy MambaTSR
"""

import sys
from pathlib import Path

print("=" * 80)
print("🔍 MambaTSR Environment Check")
print("=" * 80)

# 1. Check Python version
print("\n1. Python Version:")
print(f"   ✓ {sys.version}")

# 2. Check PyTorch
print("\n2. PyTorch:")
try:
    import torch
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   ❌ CUDA not available!")
except ImportError as e:
    print(f"   ❌ PyTorch not installed: {e}")
    sys.exit(1)

# 3. Check CUDA version match
print("\n3. CUDA Version Compatibility:")
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    if cuda_version == "12.4":
        print(f"   ✓ CUDA {cuda_version} matches system CUDA")
    else:
        print(f"   ⚠️  PyTorch CUDA {cuda_version} (check system CUDA)")
else:
    print("   ❌ CUDA not available")

# 4. Check Visual C++ Build Tools
print("\n4. Visual C++ Build Tools:")
try:
    import subprocess
    result = subprocess.run(['cl'], capture_output=True, text=True, shell=True)
    if 'Microsoft' in result.stderr:
        print("   ✓ MSVC compiler (cl.exe) found")
    else:
        print("   ❌ MSVC compiler not found")
        print("   ℹ️  Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
except Exception as e:
    print("   ❌ MSVC compiler not found or not in PATH")
    print("   ℹ️  Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

# 5. Check selective_scan
print("\n5. Selective Scan CUDA Kernel:")
try:
    import selective_scan_cuda_core
    print("   ✓ selective_scan_cuda_core imported successfully")
except ImportError:
    print("   ❌ selective_scan_cuda_core not installed")
    print("   ℹ️  Install: cd MambaTSR/kernels/selective_scan && pip install --no-build-isolation -e .")

# 6. Check other dependencies
print("\n6. Other Dependencies:")
deps = {
    'numpy': 'NumPy',
    'timm': 'PyTorch Image Models',
    'einops': 'Einops',
    'fvcore': 'FVCore',
    'torchvision': 'TorchVision',
    'tqdm': 'TQDM',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'sklearn': 'Scikit-learn'
}

missing = []
for module, name in deps.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ❌ {name}")
        missing.append(module)

if missing:
    print(f"\n   ℹ️  Install missing packages: pip install {' '.join(missing)}")

# 7. Check MambaTSR path
print("\n7. MambaTSR Repository:")
mamba_path = Path('G:/Dataset/MambaTSR')
if mamba_path.exists():
    print(f"   ✓ Found at: {mamba_path}")
    
    # Check key files
    key_files = [
        'models/VSSBlock_utils.py',
        'models/VSSBlock.py',
        'models/ConvNet.py',
        'models/vmamba.py',
        'kernels/selective_scan/setup.py'
    ]
    
    for file in key_files:
        if (mamba_path / file).exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ❌ {file} not found")
else:
    print(f"   ❌ MambaTSR not found at: {mamba_path}")

# 8. Check if can import MambaTSR components
print("\n8. MambaTSR Components:")
sys.path.insert(0, str(mamba_path))

try:
    from models.ConvNet import ConvNet
    print("   ✓ ConvNet")
except ImportError as e:
    print(f"   ❌ ConvNet: {e}")

try:
    from models.VSSBlock import VSSBlock
    print("   ✓ VSSBlock")
except ImportError as e:
    print(f"   ❌ VSSBlock: {e}")

try:
    from models.vmamba import SS2D, Mlp
    print("   ✓ SS2D, Mlp")
except ImportError as e:
    print(f"   ❌ SS2D, Mlp: {e}")

# 9. Test model creation (if selective_scan is available)
print("\n9. Model Creation Test:")
try:
    # This will only work if selective_scan is installed
    from models.VSSBlock_utils import Super_Mamba
    model = Super_Mamba(dims=3, depth=6, num_classes=39)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Super_Mamba model created successfully")
    print(f"   ✓ Total parameters: {total_params:,}")
except Exception as e:
    print(f"   ❌ Cannot create model: {e}")
    print("   ℹ️  This is expected if selective_scan is not installed yet")

# Summary
print("\n" + "=" * 80)
print("📊 Summary")
print("=" * 80)

if torch.cuda.is_available() and torch.version.cuda == "12.4":
    print("✅ PyTorch with CUDA 12.4: Ready")
else:
    print("❌ PyTorch/CUDA: Needs attention")

try:
    import selective_scan_cuda_core
    print("✅ Selective Scan Kernel: Installed")
except ImportError:
    print("⏳ Selective Scan Kernel: Needs installation (requires Build Tools)")

print("\n💡 Next Steps:")
print("1. If Build Tools is not installed:")
print("   → Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
print("2. After Build Tools installation:")
print("   → cd G:\\Dataset\\MambaTSR\\kernels\\selective_scan")
print("   → pip install --no-build-isolation -e .")
print("3. Then run the MambaTSR notebook!")

print("\n📖 See MAMBATSR_SETUP_GUIDE.md for detailed instructions")
print("=" * 80)
