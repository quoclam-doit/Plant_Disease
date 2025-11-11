#!/bin/bash
# Simple training starter that works with nohup

cd /mnt/g/Dataset

echo "=========================================="
echo "Starting MambaTSR Training"
echo "Time: $(date)"
echo "=========================================="

# Activate environment and run
/mnt/g/Dataset/.venv_wsl/bin/python train_mambatsr_plantvillage.py

echo "=========================================="
echo "Training finished at: $(date)"
echo "=========================================="
