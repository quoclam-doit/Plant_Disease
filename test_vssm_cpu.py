import sys
sys.path.insert(0, 'MambaTSR')
from models.vmamba import VSSM
import torch

print("Testing VSSM on CPU...")
model = VSSM(depths=[2,2,9,2], dims=[96,192,384,768], num_classes=38)
x = torch.randn(1, 3, 64, 64)

try:
    y = model(x)
    print(f"SUCCESS - Output shape: {y.shape}")
except Exception as e:
    print(f"FAILED: {e}")
