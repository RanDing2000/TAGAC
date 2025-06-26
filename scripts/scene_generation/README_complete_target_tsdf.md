# Complete Target TSDF Generation Scripts

这些脚本用于为ACRONYM和YCB数据集的场景文件添加`complete_target_tsdf`数据。

## 脚本说明

### 1. 单独数据集处理脚本

#### ACRONYM数据集
```bash
python scripts/scene_generation/acronym_objects/add_complete_target_tsdf.py [选项]
```

#### YCB数据集
```bash
python scripts/scene_generation/ycb_objects/add_complete_target_tsdf.py [选项]
```

### 2. 批处理脚本
```bash
python scripts/scene_generation/batch_add_complete_target_tsdf.py [选项]
```

## 参数说明

### 通用参数
- `--dataset_path`: 数据集路径（默认为标准路径）
- `--max_scenes`: 最大处理场景数（默认处理所有场景）
- `--dry_run`: 干运行模式，只分析不修改文件

### 批处理脚本特有参数
- `--datasets`: 选择处理的数据集 (`acronym`, `ycb`, `all`)
- `--acronym_path`: ACRONYM数据集路径
- `--ycb_path`: YCB数据集路径

## 使用示例

### 1. 测试模式（干运行）
```bash
# 测试ACRONYM数据集前10个场景
python scripts/scene_generation/acronym_objects/add_complete_target_tsdf.py --max_scenes 10 --dry_run

# 测试YCB数据集前10个场景
python scripts/scene_generation/ycb_objects/add_complete_target_tsdf.py --max_scenes 10 --dry_run

# 批量测试两个数据集
python scripts/scene_generation/batch_add_complete_target_tsdf.py --max_scenes 10 --dry_run
```

### 2. 实际处理
```bash
# 处理ACRONYM数据集的前100个场景
python scripts/scene_generation/acronym_objects/add_complete_target_tsdf.py --max_scenes 100

# 处理YCB数据集的前100个场景
python scripts/scene_generation/ycb_objects/add_complete_target_tsdf.py --max_scenes 100

# 批量处理两个数据集的前100个场景
python scripts/scene_generation/batch_add_complete_target_tsdf.py --max_scenes 100

# 处理所有场景（注意：这可能需要很长时间）
python scripts/scene_generation/batch_add_complete_target_tsdf.py
```

### 3. 只处理特定数据集
```bash
# 只处理ACRONYM数据集
python scripts/scene_generation/batch_add_complete_target_tsdf.py --datasets acronym

# 只处理YCB数据集
python scripts/scene_generation/batch_add_complete_target_tsdf.py --datasets ycb
```

## 处理流程

1. **读取场景文件**: 从`scenes/`目录读取`.npz`文件
2. **检查现有数据**: 如果已存在`complete_target_tsdf`则跳过
3. **提取网格数据**: 从`complete_target_mesh_vertices`和`complete_target_mesh_faces`创建trimesh对象
4. **网格清理**: 修复法线、移除退化面和重复面、移除未引用顶点
5. **生成TSDF**: 使用`mesh_to_tsdf`函数生成40x40x40的TSDF体积
6. **保存结果**: 将TSDF添加到原始场景文件中

## 输出统计

脚本会输出详细的处理统计信息：
- 总文件数
- 成功处理数
- 已有TSDF数
- 缺失网格数据数
- 错误数
- 成功率

## 注意事项

1. **备份数据**: 建议在处理前备份重要数据
2. **磁盘空间**: 确保有足够的磁盘空间存储TSDF数据
3. **处理时间**: 完整处理可能需要数小时，建议先用小批量测试
4. **错误处理**: 脚本会跳过有问题的文件并继续处理其他文件
5. **依赖项**: 需要安装trimesh、numpy、tqdm等依赖

## 验证结果

处理完成后，可以验证TSDF是否正确添加：

```python
import numpy as np

# 加载处理后的场景文件
data = np.load('path/to/scene.npz')

# 检查键
print('Keys:', list(data.keys()))
print('Has complete_target_tsdf:', 'complete_target_tsdf' in data)

# 检查TSDF形状
if 'complete_target_tsdf' in data:
    print('TSDF shape:', data['complete_target_tsdf'].shape)  # 应该是 (40, 40, 40)
```

## 故障排除

### 常见错误
1. **网格创建失败**: 通常由于顶点或面数据损坏，脚本会跳过这些文件
2. **TSDF生成失败**: 可能由于网格过于复杂或损坏，检查错误日志
3. **文件权限错误**: 确保对数据集目录有写权限

### 性能优化
- 使用`--max_scenes`参数限制处理数量进行测试
- 在SSD上运行以提高I/O性能
- 考虑并行处理多个数据集（使用不同终端） 