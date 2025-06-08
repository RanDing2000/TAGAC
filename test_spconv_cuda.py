import torch
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor

# 创建一个最小的 sparse tensor 测试 CUDA 支持
features = torch.randn(4, 16).cuda()
indices = torch.tensor([[0, 0, 0, 0],
                        [0, 1, 1, 1],
                        [0, 2, 2, 2],
                        [0, 3, 3, 3]], dtype=torch.int32).t().contiguous().cuda()

input_sp_tensor = SparseConvTensor(features, indices, spatial_shape=[10, 10, 10], batch_size=1)

conv = spconv.SparseConv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False).cuda()

output = conv(input_sp_tensor)
print("Success: CUDA is working with spconv.")
print("Output shape:", output.features.shape)
