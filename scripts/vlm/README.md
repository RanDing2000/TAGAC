# GroundingDINO + SAM Demo

这个demo展示了如何使用GroundingDINO进行文本引导的目标检测，然后使用SAM进行精确分割。

## 功能特点

- **文本引导检测**: 使用自然语言描述检测目标对象
- **精确分割**: 使用SAM生成高质量的分割掩码
- **可视化结果**: 生成带有彩色掩码的可视化图像
- **错误处理**: 完善的错误处理和用户友好的提示信息
- **命令行参数**: 支持灵活的参数配置
- **C++扩展支持**: 自动处理GroundingDINO C++扩展问题

## 环境要求

1. **Python依赖**:
   ```bash
   pip install torch torchvision opencv-python matplotlib pillow numpy requests tqdm
   ```

2. **GroundingDINO**: 已包含在 `src/GroundingDINO/` 目录中

3. **SAM**: 已包含在 `src/segment-anything/` 目录中

4. **模型权重文件**: 需要下载以下文件到指定位置
   - `groundingdino_swint_ogc.pth` - GroundingDINO权重 (默认路径: `/home/ran.ding/projects/TARGO/src/GroundingDINO/`)
   - `sam_vit_h_4b8939.pth` - SAM权重 (默认路径: `/home/ran.ding/projects/TARGO/src/GroundingDINO/`)

## 快速开始

### 1. 测试环境

首先测试所有依赖是否正确安装：

```bash
python test_imports.py
```

### 2. 修复GroundingDINO C++扩展 (如果需要)

如果遇到 `name '_C' is not defined` 错误，运行修复脚本：

```bash
python fix_groundingdino.py
```

这个脚本会：
- 尝试编译并安装GroundingDINO的C++扩展
- 如果编译失败，创建CPU fallback实现
- 确保demo可以正常运行

### 3. 下载模型权重

自动下载模型权重文件：

```bash
python download_models.py
```

或者手动下载到指定位置：
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
- SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

### 4. 运行Demo

使用默认图像：
```bash
python demo.py --text_prompt "person"
```

使用自定义图像：
```bash
python demo.py --image_path your_image.png --text_prompt "black figure"
```

## 使用方法

### 基本用法

```bash
python demo.py --image_path your_image.png --text_prompt "black figure"
```

### 完整参数

```bash
python demo.py \
    --image_path your_image.png \
    --text_prompt "black figure" \
    --output_path output_masked.png \
    --dino_checkpoint /path/to/groundingdino_swint_ogc.pth \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --device auto
```

### 参数说明

- `--image_path`: 输入图像路径 (默认: `/home/ran.ding/projects/TARGO/scripts/demo/image.png`)
- `--text_prompt`: 文本描述，用于检测目标对象 (默认: "black figure")
- `--output_path`: 输出图像保存路径 (默认: "output_masked.png")
- `--dino_checkpoint`: GroundingDINO模型权重路径 (默认: `/home/ran.ding/projects/TARGO/src/GroundingDINO/groundingdino_swint_ogc.pth`)
- `--sam_checkpoint`: SAM模型权重路径 (默认: `/home/ran.ding/projects/TARGO/src/GroundingDINO/sam_vit_h_4b8939.pth`)
- `--box_threshold`: 边界框置信度阈值 (默认: 0.3)
- `--text_threshold`: 文本置信度阈值 (默认: 0.25)
- `--device`: 使用的设备 (auto, cuda, cpu) (默认: auto)

## 示例

### 检测人物
```bash
python demo.py --text_prompt "person"
```

### 检测特定物体
```bash
python demo.py --image_path objects.jpg --text_prompt "red car"
```

### 检测多个物体
```bash
python demo.py --image_path scene.jpg --text_prompt "chair and table"
```

### 调整检测灵敏度
```bash
python demo.py --text_prompt "small object" --box_threshold 0.1 --text_threshold 0.1
```

### 强制使用CPU
```bash
python demo.py --text_prompt "person" --device cpu
```

## 输出说明

程序会输出以下信息：
- 模型加载状态
- C++扩展可用性
- 检测到的对象数量和置信度
- 掩码生成进度
- 最终保存路径

生成的图像将包含：
- 原始图像
- 彩色半透明掩码覆盖在检测到的对象上
- 标题显示文本提示和检测到的对象

## 故障排除

### 常见问题

1. **导入错误**: 运行 `python test_imports.py` 检查环境
2. **C++扩展错误**: 运行 `python fix_groundingdino.py` 修复
3. **文件不存在**: 运行 `python download_models.py` 下载模型权重
4. **CUDA错误**: 使用 `--device cpu` 强制使用CPU
5. **内存不足**: 尝试使用较小的图像或降低批处理大小
6. **检测不到对象**: 尝试调整 `--box_threshold` 和 `--text_threshold` 参数

### C++扩展问题

GroundingDINO需要编译C++扩展以获得最佳性能。如果遇到以下错误：

```
name '_C' is not defined
```

解决方案：

1. **自动修复** (推荐):
   ```bash
   python fix_groundingdino.py
   ```

2. **手动编译**:
   ```bash
   cd src/GroundingDINO
   python setup.py install
   ```

3. **使用CPU模式**:
   ```bash
   python demo.py --device cpu
   ```

### 调试模式

程序包含详细的错误信息和进度提示，如果遇到问题，请查看控制台输出。

### 环境检查

```bash
# 检查所有依赖
python test_imports.py

# 检查模型文件
ls -la /home/ran.ding/projects/TARGO/src/GroundingDINO/*.pth

# 检查默认图像
ls -la /home/ran.ding/projects/TARGO/scripts/demo/image.png

# 检查C++扩展
python -c "import groundingdino._C; print('C++ extension available')"
```

## 技术细节

- **GroundingDINO**: 使用SwinT-OGC配置，支持开放词汇目标检测
- **SAM**: 使用ViT-H模型，提供高质量分割
- **可视化**: 使用matplotlib生成高质量输出图像
- **坐标转换**: 自动处理不同坐标系之间的转换
- **路径管理**: 自动处理本地模块的导入路径
- **C++扩展**: 提供高性能的边界框操作，支持CPU fallback

## 文件结构

```
scripts/vlm/
├── demo.py                    # 主演示脚本
├── test_imports.py           # 环境测试脚本
├── download_models.py        # 模型下载脚本
├── fix_groundingdino.py      # C++扩展修复脚本
├── example_usage.py          # 使用示例脚本
├── README.md                 # 使用说明
├── CHANGELOG.md              # 变更日志
└── groundingdino_swint_ogc.pth  # GroundingDINO权重 (需要下载)
```

## 性能说明

- **GPU模式**: 需要CUDA和编译的C++扩展，性能最佳
- **CPU模式**: 使用PyTorch实现的fallback，性能较慢但兼容性好
- **内存使用**: 建议至少8GB RAM，GPU模式需要更多显存

## 许可证

请参考GroundingDINO和SAM的原始许可证。 