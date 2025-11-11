import torch
import torch.nn.functional as F

print('Testing CUDA on RTX 5060 Ti (sm_120)...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

x = torch.randn(1, 3, 32, 32).cuda()
print(f'✓ Tensor created on GPU: {x.device}')

y = F.max_pool2d(x, kernel_size=2, stride=2)
print(f'✓ max_pool2d SUCCESS! Output shape: {y.shape}')

print('\n🎉 ALL TESTS PASSED! RTX 5060 Ti works with PyTorch nightly!')
