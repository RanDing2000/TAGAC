# 添加类别标签和CLIP特征到场景数据

这个脚本用于为TARGO数据集中的场景npz文件添加类别标签和CLIP特征。

## 功能概述

该脚本会：

1. **读取场景npz文件**：从指定的scenes目录读取所有npz文件
2. **提取对象信息**：从对应的mesh_pose_dict文件中获取对象网格、位姿和缩放信息
3. **分配类别标签**：基于对象名称和预定义的类别映射分配类别标签
4. **生成CLIP特征**：使用CLIP模型为每个类别生成512维的文本特征
5. **空间分配**：将点云中的点分配给相应的对象，基于空间邻近性
6. **保存增强数据**：将原始数据与新增的标签和特征一起保存

## 新增的数据字段

处理后的npz文件将包含以下新字段：

- `points`: 合并后的点云坐标 [N, 3]
- `instance_labels`: 实例标签 [N] (0=场景对象, 1=目标对象)
- `category_labels`: 类别标签 [N] (基于预定义类别映射)
- `object_labels`: 对象标签 [N] (每个点的对象ID)
- `clip_features`: CLIP特征 [N, 512] (每个点的512维CLIP特征)

## 使用方法

### 基本用法

```bash
python scripts/scene_generation/add_category_labels_clip_feature.py
```

### 自定义参数

```bash
python scripts/scene_generation/add_category_labels_clip_feature.py \
    --scenes_dir /path/to/scenes \
    --mesh_pose_dir /path/to/mesh_pose_dict \
    --output_dir /path/to/output \
    --backup_original \
    --max_files 100
```

### 参数说明

- `--scenes_dir`: 包含场景npz文件的目录 (默认: `/usr/stud/dira/GraspInClutter/targo/data_scenes/targo_dataset/scenes`)
- `--mesh_pose_dir`: 包含mesh_pose_dict文件的目录 (默认: `/usr/stud/dira/GraspInClutter/targo/data_scenes/targo_dataset/mesh_pose_dict`)
- `--output_dir`: 输出增强文件的目录 (默认: `/usr/stud/dira/GraspInClutter/targo/data_scenes/targo_dataset/scenes_enhanced`)
- `--backup_original`: 在处理前备份原始文件
- `--max_files`: 最大处理文件数 (用于测试)

## 类别映射

脚本使用预定义的类别映射，将ACRONYM数据集中的对象映射到以下类别：

### 主要类别

1. **厨房用品**: bowl, mug, bottle, pot, vase
2. **餐具**: fork, ladle
3. **工具**: hammer, screwdriver, wrench, pliers, scissors, stapler
4. **办公用品**: pen, eraser, ruler, calculator
5. **电子产品**: phone, remote, keyboard, mouse, headphones, speaker
6. **书籍纸张**: book, folder
7. **玩具**: toy
8. **植物**: plant
9. **食物**: food
10. **服装**: clothing
11. **工具**: tool
12. **家具**: furniture
13. **电子产品**: electronics

### 类别映射文件

脚本会生成两个映射文件：

1. `category_mapping.json`: 类别索引到类别名称的映射
2. `object_category_mapping.json`: 对象名称到类别的映射

## 技术细节

### CLIP特征提取

- 使用CLIP ViT-B/32模型
- 为每个类别生成"a {category}"格式的文本特征
- 特征维度：512维
- 支持GPU加速

### 空间分配算法

1. 将点云坐标从[-0.5, 0.5]转换到世界坐标系
2. 对每个对象网格应用变换（缩放+位姿）
3. 计算对象的边界框
4. 将边界框内的点分配给该对象
5. 添加容差以提高鲁棒性

### 错误处理

- 自动跳过格式不正确的文件
- 详细的错误日志和堆栈跟踪
- 支持部分处理（即使某些文件失败）

## 输出示例

处理后的文件结构：

```
scenes_enhanced/
├── fffe85b2a80348ee94e25198c7c6169e_c_3.npz
├── fffe85b2a80348ee94e25198c7c6169e_c_2.npz
├── ...
├── category_mapping.json
└── object_category_mapping.json
```

每个npz文件包含：

```python
# 原始数据
grid_scene: [40, 40, 40]  # 场景体素网格
grid_targ: [40, 40, 40]   # 目标体素网格
pc_depth_scene_no_targ: [N, 3]  # 场景点云
pc_depth_targ: [M, 3]     # 目标点云

# 新增数据
points: [N+M, 3]          # 合并后的点云
instance_labels: [N+M]    # 实例标签
category_labels: [N+M]    # 类别标签
object_labels: [N+M]      # 对象标签
clip_features: [N+M, 512] # CLIP特征
```

## 性能考虑

- **内存使用**: 处理大量文件时注意内存使用
- **GPU内存**: CLIP模型需要GPU内存，确保有足够空间
- **处理时间**: 每个文件处理时间取决于点云大小和对象数量
- **存储空间**: 增强文件比原始文件大约大50-100%

## 故障排除

### 常见问题

1. **CLIP模型加载失败**
   - 确保已安装clip包：`pip install clip`
   - 检查CUDA可用性

2. **内存不足**
   - 使用`--max_files`参数分批处理
   - 增加系统内存或使用更大GPU

3. **文件格式错误**
   - 检查npz文件完整性
   - 确保mesh_pose_dict文件存在

4. **类别映射问题**
   - 检查对象名称格式
   - 更新`get_category_mapping()`函数

### 调试模式

添加详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 自定义类别映射

修改`get_category_mapping()`函数以支持自定义类别：

```python
def get_category_mapping():
    # 添加自定义类别
    custom_mapping = {
        "custom_category": ["object1", "object2", "object3"]
    }
    return custom_mapping
```

### 支持其他数据集

修改`extract_object_category_from_path()`函数以支持不同的命名约定：

```python
def extract_object_category_from_path(mesh_path):
    # 根据数据集调整提取逻辑
    filename = os.path.basename(mesh_path)
    # 自定义提取逻辑
    return category
``` 