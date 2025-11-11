#!/bin/bash
# Quick start script for MambaTSR training on PlantVillage
# RTX 5060 Ti (sm_120) Compatible

echo "=============================================="
echo "MambaTSR Training - PlantVillage Dataset"
echo "RTX 5060 Ti (sm_120) - Ready!"
echo "=============================================="
echo ""

# Check GPU
echo "Checking GPU..."
/mnt/g/Dataset/.venv_wsl/bin/python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'Compute Capability: {torch.cuda.get_device_capability()}')
"

echo ""
echo "=============================================="
echo "Starting Training..."
echo "=============================================="
echo ""

# Run training
/mnt/g/Dataset/.venv_wsl/bin/python /mnt/g/Dataset/train_mambatsr_plantvillage.py

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
