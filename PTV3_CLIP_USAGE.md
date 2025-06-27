# PTV3 CLIP Model Usage Guide

## 概述

PTV3 CLIP 是一个基于 PointTransformerV3 的模型，它集成了 CLIP 特征来增强对点云语义的理解。该模型基于 `ptv3_scene` 模型，但增加了对 CLIP 特征的支持。

## 模型架构

- **基础架构**: PointTransformerV3
- **输入格式**: `[B, N, 516]` 其中：
  - `[..., :3]`: 3D 坐标 (xyz)
  - `[..., 3]`: 二进制标签 (1=目标, 0=遮挡物)
  - `[..., 4:516]`: CLIP 特征 (512维)
- **输出**: 密集网格特征，用于下游的抓取预测

## 文件结构

```
src/
├── transformer/
│   └── ptv3_clip_model.py          # PTV3 CLIP 模型定义
├── vgn/
│   ├── networks.py                  # 网络配置 (包含 Ptv3ClipNet)
│   ├── dataset_voxel.py            # 数据集类 (包含 DatasetVoxel_PTV3_Clip)
│   └── ConvONets/conv_onet/
│       ├── config.py               # 模型配置 (包含 get_model_ptv3_clip)
│       └── models/__init__.py      # 模型类 (支持 ptv3_clip)
scripts/
└── train/
    └── train_ptv3_clip.py          # 训练脚本
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 12.1.0 (推荐)
- 其他依赖项与 PointTransformerV3 相同

## 使用方法

### 1. 基本模型创建

```python
from src.vgn.networks import get_network

# 创建 PTV3 CLIP 模型
model = get_network("ptv3_clip")
```

### 2. 数据准备

模型期望输入数据格式为 `[B, N, 516]`：

```python
import torch

# 创建示例数据
batch_size = 2
num_points = 1024

# 输入数据
input_data = torch.randn(batch_size, num_points, 516)

# 设置坐标范围 [-0.5, 0.5]
input_data[:, :, :3] = input_data[:, :, :3] * 0.5

# 设置二进制标签
input_data[:, :, 3] = torch.randint(0, 2, (batch_size, num_points)).float()

# CLIP 特征 (512维)
# input_data[:, :, 4:516] 应该包含预计算的 CLIP 特征
```

### 3. 前向传播

```python
model.eval()
with torch.no_grad():
    output = model(input_data)
print(f"输出形状: {output.shape}")
```

### 4. 训练

使用提供的训练脚本：

```bash
# 基本训练
python scripts/train/train_ptv3_clip.py \
    --dataset /path/to/dataset \
    --dataset_raw /path/to/raw_dataset \
    --net ptv3_clip \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4

# 使用 CLIP 特征
python scripts/train/train_ptv3_clip.py \
    --dataset /path/to/dataset \
    --dataset_raw /path/to/raw_dataset \
    --net ptv3_clip \
    --clip_feature_dir /path/to/clip_features \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4
```

## CLIP 特征集成

### 1. 生成 CLIP 特征

首先使用提供的脚本生成 CLIP 特征：

```bash
python scripts/scene_generation/targo_objects/add_category_labels_clip_feature.py \
    --scenes_dir /path/to/scenes \
    --output_dir /path/to/clip_features
```

### 2. CLIP 特征格式

CLIP 特征文件应保存为 `.npz` 格式，包含：

```python
# 文件: {scene_id}_clip_features.npz
{
    'target_clip': np.array,  # [N_target, 512] 目标点云的 CLIP 特征
    'scene_clip': np.array,   # [N_scene, 512] 场景点云的 CLIP 特征
}
```

### 3. 数据集集成

`DatasetVoxel_PTV3_Clip` 类会自动加载 CLIP 特征：

```python
from src.vgn.dataset_voxel import DatasetVoxel_PTV3_Clip

dataset = DatasetVoxel_PTV3_Clip(
    root="/path/to/dataset",
    raw_root="/path/to/raw_dataset",
    model_type="ptv3_clip",
    clip_feature_dir="/path/to/clip_features"
)
```

## 配置参数

### 模型配置

在 `src/vgn/networks.py` 中的 `Ptv3ClipNet()` 函数：

```python
cfg = {
    'model_type': 'ptv3_clip',
    'd_model': 64,
    'encoder': 'voxel_simple_local_without_3d',
    'decoder': 'simple_local',
    'c_dim': 64,
    # ... 其他参数
}
```

### 训练参数

主要训练参数：

- `--net ptv3_clip`: 指定模型类型
- `--clip_feature_dir`: CLIP 特征目录
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--use_wandb`: 是否使用 wandb 记录

## 测试

运行测试脚本验证模型功能：

```bash
python test_ptv3_clip_model.py
```

## 与 ptv3_scene 的区别

| 特性 | ptv3_scene | ptv3_clip |
|------|------------|-----------|
| 输入维度 | [B, N, 4] | [B, N, 516] |
| 特征类型 | 几何 + 标签 | 几何 + 标签 + CLIP |
| CLIP 支持 | ❌ | ✅ |
| 语义理解 | 基础 | 增强 |

## 故障排除

### 常见问题

1. **CLIP 特征未找到**
   - 确保 `clip_feature_dir` 路径正确
   - 检查 CLIP 特征文件是否存在

2. **内存不足**
   - 减少 `batch_size`
   - 减少 `num_points`

3. **CUDA 错误**
   - 确保使用正确的 CUDA 版本 (12.1.0)
   - 检查 GPU 内存

### 调试

启用调试模式：

```bash
python scripts/train/train_ptv3_clip.py \
    --debug \
    --net ptv3_clip \
    # ... 其他参数
```

## 性能优化

1. **数据加载优化**
   - 使用 SSD 存储
   - 调整 `num_workers`

2. **模型优化**
   - 使用混合精度训练
   - 启用梯度累积

3. **内存优化**
   - 使用梯度检查点
   - 减少中间特征存储

## 扩展

### 自定义 CLIP 特征

可以修改 `DatasetVoxel_PTV3_Clip` 类来支持自定义 CLIP 特征格式：

```python
def load_custom_clip_features(self, scene_id):
    # 实现自定义 CLIP 特征加载逻辑
    pass
```

### 添加新的特征类型

在 `PointTransformerV3CLIPModel.prepare_point_data()` 中添加新特征：

```python
# 添加新特征维度
new_features = torch.randn(batch_size, num_points, new_feature_dim)
feats = torch.cat([coords, label_onehot, clip_features, new_features], dim=-1)
```

## 贡献

欢迎提交问题和改进建议！ 