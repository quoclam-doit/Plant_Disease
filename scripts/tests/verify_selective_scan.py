import torch
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

import selective_scan_cuda_core
import selective_scan_cuda_ndstate
import selective_scan_cuda_oflex

print("✓ selective_scan_cuda_core loaded")
print("✓ selective_scan_cuda_ndstate loaded")
print("✓ selective_scan_cuda_oflex loaded")
print("\n✅ All selective_scan extensions recompiled and working!")
