# Random Texture Scene Generation

这个脚本基于原始的 `generate_kitchen_dataset_gso_scene.py`，添加了随机纹理功能，可以为每个物体应用随机的材质纹理。

## 功能特点

1. **自动纹理发现**: 自动扫描 `/home/ran.ding/projects/TARGO/data/textures/` 目录中的可用纹理
2. **随机纹理应用**: 为每个物体随机选择并应用纹理
3. **纹理缓存**: 缓存已加载的纹理以提高性能
4. **多种纹理支持**: 支持漫反射、法线、粗糙度、金属度等纹理类型
5. **向后兼容**: 当没有纹理可用时，回退到原始的颜色编码

## 使用方法

### 1. 下载纹理（如果还没有）

```bash
cd /home/ran.ding/projects/TARGO
python data/download_cc_textures.py
```

### 2. 测试纹理功能

```bash
cd scripts/scene_generation/kitchen_gso_objects
python test_random_texture.py
```

### 3. 生成带随机纹理的场景

```bash
# 基本用法
python generate_kitchen_dataset_gso_scene_random_texture.py --num-scenes 10

# 指定输出目录
python generate_kitchen_dataset_gso_scene_random_texture.py \
    --root /path/to/output \
    --num-scenes 50 \
    --object-set mess_kitchen/train

# 禁用纹理（回退到原始颜色编码）
python generate_kitchen_dataset_gso_scene_random_texture.py \
    --no-textures \
    --num-scenes 10

# 指定自定义纹理目录
python generate_kitchen_dataset_gso_scene_random_texture.py \
    --texture-root /path/to/custom/textures \
    --num-scenes 10
```

## 命令行参数

### 新增的纹理相关参数

- `--texture-root`: 纹理根目录路径（默认: `/home/ran.ding/projects/TARGO/data/textures`）
- `--no-textures`: 禁用纹理应用，使用原始颜色编码

### 其他参数（与原始脚本相同）

- `--root`: 输出根目录
- `--num-scenes`: 生成的场景数量
- `--object-set`: 物体集合（`mess_kitchen/train` 或 `mess_kitchen/test`）
- `--enable-visualization`: 启用可视化
- `--vis-depth`: 可视化深度图
- `--vis-segmentation`: 可视化分割图
- `--save-raw-data`: 保存原始数据

## 输出文件

脚本会生成以下文件：

1. **GLB文件**: 带纹理的3D场景文件 (`g1b_files/{scene_id}_combined.glb`)
2. **元数据**: 包含纹理信息的JSON文件 (`g1b_files/{scene_id}_combined_metadata.json`)
3. **渲染图像**: 使用PyVista生成的场景渲染图
4. **可视化**: 深度图和分割图（如果启用）
5. **原始数据**: 用于分析的numpy数组和JSON文件

## 纹理格式支持

脚本支持以下纹理文件命名约定：

- **漫反射纹理**: `*_Diffuse*.jpg/png`
- **法线纹理**: `*_Normal*.jpg/png`
- **粗糙度纹理**: `*_Roughness*.jpg/png`
- **金属度纹理**: `*_Metallic*.jpg/png`
- **环境光遮蔽**: `*_AO*.jpg/png`

## 纹理目录结构

预期的纹理目录结构：

```
/home/ran.ding/projects/TARGO/data/textures/
├── TextureName1/
│   ├── TextureName1_2K-JPG_Diffuse.jpg
│   ├── TextureName1_2K-JPG_Normal.jpg
│   ├── TextureName1_2K-JPG_Roughness.jpg
│   └── TextureName1_2K-JPG_Metallic.jpg
├── TextureName2/
│   ├── TextureName2_2K-JPG_Diffuse.jpg
│   └── ...
└── ...
```

## 性能优化

1. **纹理缓存**: 已加载的纹理会被缓存，避免重复加载
2. **批量处理**: 支持批量生成多个场景
3. **可选可视化**: 可以禁用可视化以提高性能

## 故障排除

### 常见问题

1. **没有找到纹理**
   - 确保已运行 `python data/download_cc_textures.py`
   - 检查 `TEXTURE_ROOT` 路径是否正确

2. **纹理加载失败**
   - 检查纹理文件是否完整
   - 确保文件权限正确

3. **GLB导出失败**
   - 脚本会自动回退到PLY格式
   - 检查磁盘空间是否充足

### 调试

运行测试脚本检查功能：

```bash
python test_random_texture.py
```

查看详细日志：

```bash
python generate_kitchen_dataset_gso_scene_random_texture.py --num-scenes 1 2>&1 | tee debug.log
```

## 示例输出

### 元数据文件示例

```json
{
  "scene_id": "a1b2c3d4e5f6",
  "object_count": 3,
  "objects": {
    "1": "path/to/mesh1.obj",
    "2": "path/to/mesh2.obj",
    "3": "path/to/mesh3.obj"
  },
  "applied_textures": {
    "1": "WoodChips001",
    "2": "MetalWalkway005",
    "3": "Ground004"
  },
  "textures_available": 95
}
```

### 控制台输出示例

```
============================================================
Scene Generation with Random Textures
============================================================
Output directory: /path/to/output
Number of scenes: 10
Texture root: /home/ran.ding/projects/TARGO/data/textures
Textures enabled: True
Visualization enabled: True
  - Depth visualization: True
  - Segmentation visualization: True
  - Save raw data: True
============================================================
Found 95 texture directories
Applied texture 'WoodChips001' to mesh
Applied texture 'MetalWalkway005' to mesh
Applied texture 'Ground004' to mesh
Saved combined GLB with random textures: /path/to/output/g1b_files/a1b2c3d4e5f6_combined.glb
```

## 技术细节

### 纹理应用流程

1. **纹理发现**: 扫描纹理目录，找到包含纹理文件的子目录
2. **材质创建**: 为每个纹理创建PBR材质对象
3. **随机选择**: 为每个物体随机选择纹理
4. **材质应用**: 将材质应用到mesh的visual属性
5. **缓存管理**: 缓存已加载的材质以提高性能

### 兼容性

- 支持trimesh.Scene和trimesh.Trimesh对象
- 自动处理单mesh和多mesh场景
- 向后兼容原始颜色编码
- 支持GLB和PLY导出格式

## 更新日志

- **v1.0**: 初始版本，支持基本随机纹理功能
- 支持纹理发现和缓存
- 支持多种纹理类型
- 添加测试脚本和文档
