#!/usr/bin/env python3
"""Test MambaTSR model forward pass on RTX 5060 Ti"""

import sys
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR')
sys.path.insert(0, '/mnt/g/Dataset/MambaTSR/kernels/selective_scan')

import torch
import torch.nn as nn

print("=" * 60)
print("Testing MambaTSR Model on RTX 5060 Ti (sm_120)")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Test selective_scan imports
print("\n" + "=" * 60)
print("Step 1: Testing selective_scan kernels")
print("=" * 60)

try:
    import selective_scan_cuda_core
    print("✓ selective_scan_cuda_core loaded")
    import selective_scan_cuda_ndstate
    print("✓ selective_scan_cuda_ndstate loaded")
    import selective_scan_cuda_oflex
    print("✓ selective_scan_cuda_oflex loaded")
except Exception as e:
    print(f"✗ Failed to import selective_scan: {e}")
    exit(1)

# Test MambaTSR model import and initialization
print("\n" + "=" * 60)
print("Step 2: Loading MambaTSR model")
print("=" * 60)

try:
    from models.vmamba import VSSM
    print("✓ VSSM model imported successfully")
    
    # Create a small model for testing
    print("\nCreating VSSM model (tiny configuration)...")
    model = VSSM(
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=True,
        forward_type="v2",
        mlp_ratio=4.0,
        downsample_version="v2",
        patchembed_version="v1",
    )
    
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test model forward pass on CPU first
print("\n" + "=" * 60)
print("Step 3: Testing forward pass on CPU")
print("=" * 60)

try:
    model.eval()
    batch_size = 2
    img_size = 224
    
    print(f"Creating random input: [{batch_size}, 3, {img_size}, {img_size}]")
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    print("Running forward pass on CPU...")
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ CPU forward pass SUCCESS!")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ CPU forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test model forward pass on GPU
print("\n" + "=" * 60)
print("Step 4: Testing forward pass on GPU (RTX 5060 Ti)")
print("=" * 60)

if not torch.cuda.is_available():
    print("✗ CUDA not available, skipping GPU test")
    exit(1)

try:
    print("Moving model to GPU...")
    model = model.cuda()
    print("✓ Model moved to GPU")
    
    print(f"Creating random input on GPU: [{batch_size}, 3, {img_size}, {img_size}]")
    x = torch.randn(batch_size, 3, img_size, img_size).cuda()
    
    print("Running forward pass on GPU...")
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ GPU forward pass SUCCESS!")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"✗ GPU forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("✓ MambaTSR model works correctly on RTX 5060 Ti (sm_120)")
print("✓ Ready to train on PlantVillage dataset!")
print("=" * 60)
