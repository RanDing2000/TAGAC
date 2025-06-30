# PTVis 安装和使用说明

## 概述

PTVis 是一个用于3D网格渲染的Python包，用于将colored_scene_mesh渲染成图片。

## 安装

### 方法1: 使用pip安装
```bash
pip install ptvis
```

### 方法2: 从源码安装
```bash
git clone https://github.com/your-ptvis-repo/ptvis.git
cd ptvis
pip install -e .
```

### 方法3: 使用conda安装
```bash
conda install -c conda-forge ptvis
```

## 验证安装

运行以下命令验证ptvis是否正确安装：

```bash
python -c "import ptvis; print('PTVis version:', ptvis.__version__)"
```

## 在TARGO项目中的使用

### 1. 自动渲染

当使用ptv3_scene模型进行推理时，如果设置了 `visualize=True`，系统会自动：

1. 生成colored_scene_mesh（包含抓取质量的颜色信息）
2. 保存为 `demo/ptv3_scene_affordance_visual.obj`
3. 使用ptvis渲染为 `demo/ptv3_scene_affordance_visual.png`

### 2. 手动测试

运行测试脚本验证ptvis功能：

```bash
python scripts/scene_generation/targo_objects/test_ptvis_rendering.py
```

### 3. 代码示例

```python
from src.vgn.detection_ptv3_implicit import render_colored_scene_mesh_with_ptvis
import trimesh

# 加载colored scene mesh
mesh = trimesh.load("demo/ptv3_scene_affordance_visual.obj")

# 渲染为图片
success = render_colored_scene_mesh_with_ptvis(
    mesh,
    output_path="demo/rendered_image.png",
    width=800,
    height=600,
    camera_distance=0.5
)

if success:
    print("渲染成功！")
else:
    print("渲染失败！")
```

## 渲染参数说明

- `colored_scene_mesh`: 包含颜色信息的trimesh对象
- `output_path`: 输出图片路径
- `width`: 图片宽度（默认800）
- `height`: 图片高度（默认600）
- `camera_distance`: 相机距离场景中心的距离（默认0.5）

## 输出文件

- `demo/ptv3_scene_affordance_visual.obj`: 包含颜色信息的3D网格文件
- `demo/ptv3_scene_affordance_visual.png`: 渲染的2D图片文件

## 颜色说明

渲染的图片中，颜色表示抓取质量：
- 红色：高质量抓取区域
- 颜色越深：抓取质量越高
- 颜色越浅：抓取质量越低

## 故障排除

### 1. 导入错误
如果遇到 `ImportError: No module named 'ptvis'`，请确保正确安装了ptvis包。

### 2. 渲染失败
如果渲染失败，请检查：
- 网格文件是否有效
- 输出目录是否有写入权限
- 相机参数是否合理

### 3. 性能问题
如果渲染速度较慢，可以：
- 降低图片分辨率
- 简化网格复杂度
- 调整相机距离

## 相关文件

- `src/vgn/detection_ptv3_implicit.py`: 主要的渲染实现
- `scripts/scene_generation/targo_objects/test_ptvis_rendering.py`: 测试脚本
- `docs/ptvis_installation.md`: 本说明文档 