#!/usr/bin/env python3
"""Simple test for MambaTSR imports"""

import sys
import os

# Change to MambaTSR directory
os.chdir('/mnt/g/Dataset/MambaTSR')
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR')
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR/kernels/selective_scan')

print("Working directory:", os.getcwd())
print("Python path:", sys.path[:3])

import torch
print(f"\n✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# Import selective_scan
try:
    import selective_scan_cuda_core
    print("✓ selective_scan_cuda_core loaded")
except Exception as e:
    print(f"✗ selective_scan_cuda_core: {e}")

# Try importing MambaTSR components step by step
print("\nTesting MambaTSR imports...")

try:
    print("  Importing models package...")
    import models
    print(f"  ✓ models package: {models.__file__}")
except Exception as e:
    print(f"  ✗ models package failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("  Importing models.vmamba...")
    from models import vmamba
    print(f"  ✓ models.vmamba: {vmamba.__file__}")
except Exception as e:
    print(f"  ✗ models.vmamba failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("  Importing VSSM class...")
    from models.vmamba import VSSM
    print(f"  ✓ VSSM class imported")
except Exception as e:
    print(f"  ✗ VSSM import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
