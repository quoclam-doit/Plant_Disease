#!/usr/bin/env python3
"""Direct import test bypassing __init__.py"""

import sys
import os

os.chdir('/mnt/g/Dataset/MambaTSR')
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR')
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR/kernels/selective_scan')

import torch
print(f"✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# Import selective_scan
import selective_scan_cuda_core
print("✓ selective_scan_cuda_core loaded")

# Try direct import from vmamba.py file
print("\nDirect import from models/vmamba.py...")
try:
    # Import spec_utils to load the module without __init__
    import importlib.util
    spec = importlib.util.spec_from_file_location("vmamba_module", "/mnt/g/Dataset/MambaTSR/models/vmamba.py")
    vmamba_module = importlib.util.module_from_spec(spec)
    
    print("Loading vmamba module...")
    spec.loader.exec_module(vmamba_module)
    
    print(f"✓ vmamba module loaded")
    print(f"  Has VSSM: {hasattr(vmamba_module, 'VSSM')}")
    
    if hasattr(vmamba_module, 'VSSM'):
        VSSM = vmamba_module.VSSM
        print(f"  VSSM class: {VSSM}")
        
        # Try creating a small model
        print("\nCreating tiny VSSM model...")
        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=10,
            depths=[2, 2],
            dims=[96, 192],
        )
        print(f"✓ Model created! Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # MambaTSR requires GPU (selective_scan kernels are CUDA-only)
        if not torch.cuda.is_available():
            print("\n✗ CUDA not available! MambaTSR requires GPU.")
        else:
            print("\nTesting GPU forward pass on RTX 5060 Ti...")
            model = model.cuda()
            model.eval()
            x = torch.randn(2, 3, 224, 224).cuda()
            
            # Record memory before forward
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / 1024**2
            
            with torch.no_grad():
                y = model(x)
            
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            current_mem = torch.cuda.memory_allocated() / 1024**2
            
            print(f"✓ GPU forward SUCCESS! Output shape: {y.shape}")
            print(f"  Memory before: {start_mem:.1f} MB")
            print(f"  Memory peak: {peak_mem:.1f} MB")
            print(f"  Memory after: {current_mem:.1f} MB")
            print(f"  Memory used: {peak_mem - start_mem:.1f} MB")
            print("\n🎉 MambaTSR works on RTX 5060 Ti (sm_120)!")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
