# Binary Label Feature Integration for PointTransformerV3

## 概述

为了让PointTransformerV3模型能够区分target和scene点，我们在dataset中添加了binary label feature。每个点云现在包含4维信息：`[x, y, z, binary_label]`，其中binary_label用于区分target点（1）和scene点（0）。

## 修改内容

### 1. Dataset修改 (`src/vgn/dataset_voxel.py`)

#### DatasetVoxel_Target类的修改：

```python
# 添加binary labels
targ_labels = np.ones((targ_pc.shape[0], 1), dtype=np.float32)  # Target points labeled as 1
scene_labels = np.zeros((scene_pc.shape[0], 1), dtype=np.float32)  # Scene points labeled as 0

# 连接点云和labels [x, y, z, binary_label]
targ_pc_with_labels = np.concatenate([targ_pc, targ_labels], axis=1)  # [N, 4]
scene_pc_with_labels = np.concatenate([scene_pc, scene_labels], axis=1)  # [N, 4]
```

#### 不同模型类型的数据格式：

- **targo_ptv3**: 返回 `(scene_pc_with_labels, targ_pc_with_labels)`
- **ptv3_scene**: 返回 `(combined_pc_with_labels)` - 合并scene和target点云

### 2. PointTransformerV3SceneModel修改 (`src/transformer/ptv3_scene_model.py`)

#### prepare_point_data方法的增强：

```python
def prepare_point_data(self, points):
    """
    现在支持4D输入: [x, y, z, binary_label] 
    其中binary_label指示target(1) vs scene(0)
    """
    # 提取binary labels（第4个通道）
    if feature_dim >= 4:
        binary_labels = points[:, :, 3:4]  # binary label channel
    
    # 智能feature padding策略
    if target_feat_dim == 6 and feats_data.shape[-1] == 4:
        # 常见情况：将[xyz, binary] pad到6D，使用xy作为padding
        padding = coords_data[:, :, :2]  # Use xy as padding
```

#### 固定Grid Size设置：

- 设置 `sparse_shape = [40, 40, 40]` 确保输出固定尺寸
- 坐标归一化到 `[-0.5, 0.5]` 范围
- `grid_size = 1.0/40` 确保40x40x40的voxel grid

### 3. 训练脚本修改 (`scripts/train_targo_ptv3.py`)

#### prepare_batch函数更新：

```python
def prepare_batch(batch, device, model_type="targo_ptv3"):
    """现在处理包含binary labels的4D点云数据"""
    targ_pc = targ_pc.float().to(device)  # Now [B, N, 4] with binary labels
    scene_pc = scene_pc.float().to(device)  # Now [B, N, 4] with binary labels
    
    if model_type == "ptv3_scene":
        # scene_pc包含scene点(label=0)和target点(label=1)
        return scene_pc, (label, rotations, width), pos
```

## 使用方法

### 1. 模型初始化

```python
# 创建支持binary labels的模型
model = PointTransformerV3SceneModel(in_channels=6)  # 支持6维特征
```

### 2. 数据格式

输入点云现在是4维的：
```python
# 输入格式: [batch_size, num_points, 4]
# 其中第4维是binary label: 0=scene点, 1=target点
points = torch.tensor([
    [[x1, y1, z1, 0],   # scene点
     [x2, y2, z2, 1],   # target点
     ...]
])
```

### 3. 模型输出

```python
grid_feat = model(points)  # 输出形状: [BS, 64, 40, 40, 40]
```

## 数据流程

1. **DatasetVoxel_Target**: 
   - 读取原始点云数据
   - 为target点添加label=1，scene点添加label=0
   - 返回4维点云数据

2. **PointTransformerV3SceneModel**:
   - `prepare_point_data()`: 处理4维输入，智能padding到6维
   - 坐标归一化和grid size设置
   - 返回固定尺寸的grid feature

3. **Training Scripts**:
   - `prepare_batch()`: 处理4维点云batch数据
   - 根据模型类型选择适当的数据格式

## 优势

1. **语义信息**: Binary label提供了点的语义信息（target vs scene）
2. **固定输出尺寸**: 确保grid feature为固定的40x40x40
3. **兼容性**: 与现有训练流程兼容
4. **灵活性**: 支持不同模型架构（targo_ptv3, ptv3_scene）

## 测试

运行测试脚本验证功能：
```bash
python test_binary_labels.py
```

该脚本将验证：
- 4维点云输入的处理
- 固定尺寸输出的生成
- 不同模型模式的兼容性 