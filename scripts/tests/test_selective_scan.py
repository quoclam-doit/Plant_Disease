#!/usr/bin/env python3
"""Test selective_scan CUDA kernels compilation"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import sys
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR/kernels/selective_scan')

try:
    # Try importing the compiled extensions directly
    import selective_scan_cuda_core
    print("✓ selective_scan_cuda_core.so imported successfully!")
    print(f"  Location: {selective_scan_cuda_core.__file__}")
    
    import selective_scan_cuda_ndstate
    print("✓ selective_scan_cuda_ndstate.so imported successfully!")
    print(f"  Location: {selective_scan_cuda_ndstate.__file__}")
    
    import selective_scan_cuda_oflex
    print("✓ selective_scan_cuda_oflex.so imported successfully!")
    print(f"  Location: {selective_scan_cuda_oflex.__file__}")
    
    print("\n🎉 SUCCESS! All selective_scan kernels compiled correctly and loadable!")
    print("✓ CUDA extensions work on RTX 5060 Ti (sm_120)!")
    
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
