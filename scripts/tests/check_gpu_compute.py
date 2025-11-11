import torch

print(f"GPU: {torch.cuda.get_device_name()}")
cap = torch.cuda.get_device_capability()
print(f"Compute capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")
print(f"PyTorch compiled with CUDA arch list: {torch.cuda.get_arch_list()}")
