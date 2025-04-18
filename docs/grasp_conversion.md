# 抓取格式转换工具

本文档介绍了在 VGN、AnyGrasp 和 FGC-Grasp 格式之间进行抓取转换的工具和方法。

## 支持的格式

### VGN 格式

VGN 使用 `vgn.grasp.Grasp` 类表示抓取：

```python
class Grasp(object):
    def __init__(self, pose, width):
        self.pose = pose  # Transform 对象，包含旋转和平移
        self.width = width  # 夹爪宽度
```

其中 `pose` 是一个 `Transform` 对象，包含旋转和平移信息。

### AnyGrasp/GraspNet 格式

AnyGrasp 使用 `graspnetAPI` 中的 `GraspGroup` 和 `Grasp` 类表示抓取：

```python
class Grasp:
    def __init__(self, rotation_matrix, translation, width, score):
        self.rotation_matrix = rotation_matrix  # 3x3 旋转矩阵
        self.translation = translation  # 3D 位置向量
        self.width = width  # 夹爪宽度
        self.score = score  # 抓取质量得分
```

### FGC-Grasp 格式

FGC-Grasp 使用与 AnyGrasp 相似的格式表示抓取。

## 转换函数

我们提供了以下转换函数：

### 1. AnyGrasp/FGC-Grasp 转 VGN

```python
from vgn.grasp_conversion import anygrasp_to_vgn, fgc_to_vgn

# 转换 AnyGrasp 抓取到 VGN 格式
vgn_grasps, scores = anygrasp_to_vgn(
    grasp_group,         # AnyGrasp/GraspNet 格式的 GraspGroup
    extrinsic,           # 相机外参矩阵（相机到世界的转换）
    workspace_size=None, # 可选：工作空间大小限制，用于过滤抓取
    gripper_offset=np.array([0, 0, -0.02])  # 可选：Franka 夹爪偏移量
)

# 转换 FGC-Grasp 抓取到 VGN 格式（接口相同）
vgn_grasps, scores = fgc_to_vgn(
    grasp_group,
    extrinsic,
    workspace_size=None,
    gripper_offset=np.array([0, 0, -0.02])
)
```

### 2. VGN 转 AnyGrasp/GraspNet

```python
from vgn.grasp_conversion import vgn_to_anygrasp

# 转换 VGN 抓取到 AnyGrasp/GraspNet 格式
anygrasp_grasps = vgn_to_anygrasp(
    vgn_grasps,  # VGN Grasp 对象列表
    scores,      # 抓取得分列表
    extrinsic    # 相机外参矩阵（世界到相机的转换）
)
```

## 转换细节

转换过程中处理了以下关键问题：

1. **坐标系转换**：
   - AnyGrasp/GraspNet 在相机坐标系中表示抓取
   - VGN 在世界坐标系中表示抓取
   - 使用相机外参矩阵进行坐标系之间的转换

2. **方向调整**：
   - 在 Y 轴上应用 90 度旋转，以匹配不同格式之间的方向约定
   - `grasp_R = Rotation.from_matrix(grasp.rotation_matrix) * Rotation.from_euler('Y', np.pi/2)`

3. **夹爪偏移**：
   - 应用固定偏移量以适应 Franka 夹爪差异
   - 默认偏移量为沿抓取框架的 -Z 方向 2cm

4. **工作空间过滤**：
   - 可选地过滤超出工作空间边界的抓取

## 示例用法

参见 `examples/grasp_conversion_example.py` 获取完整的使用示例：

```bash
# 激活环境
conda activate targo

# 加载所需模块
module load compiler/gcc-8.3
module load cuda/11.3.0

# 运行示例
python examples/grasp_conversion_example.py
```

## 依赖项

- `numpy`
- `scipy`
- `vgn` (本项目)
- `graspnetAPI` (用于 AnyGrasp 格式，可选) 