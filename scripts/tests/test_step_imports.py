#!/usr/bin/env python3
"""Test imports step by step"""

import sys
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR')
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR/kernels/selective_scan')

print("1. torch...")
import torch
print("  OK")

print("2. torch.nn...")
import torch.nn as nn
print("  OK")

print("3. einops...")
from einops import rearrange
print("  OK")

print("4. timm...")
from timm.models.layers import DropPath
print("  OK")

print("5. fvcore...")
try:
    from fvcore.nn import FlopCountAnalysis
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")

print("6. selective_scan...")
try:
    import selective_scan_cuda_core
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nAll basic imports OK!")
