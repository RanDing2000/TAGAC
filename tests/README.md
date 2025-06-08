# TARGO++ 测试文件集合

这个目录包含了项目开发过程中创建的各种测试脚本，用于验证代码功能、调试问题以及确保系统正常运行。

## 文件分类

### 数据集相关测试
- `test_dataset_error_handling.py` - 验证DatasetVoxel_Target中的点云错误处理机制
- `test_visualization.py` - 测试数据可视化功能

### 训练相关测试
- `test_simple.py` - 简单的训练流程测试
- `test_training_validation.py` - 训练和验证流程综合测试

### 验证评估测试
- `test_validation_direct.py` - 直接验证功能测试
- `test_validation_simple.py` - 简化的验证测试
- `test_validation_monotonic_steps.py` - 验证单调步长测试
- `test_validation_upload_force.py` - 强制验证上传测试
- `test_ycb_only_validation.py` - 仅YCB数据集验证测试

### Wandb集成测试
- `test_wandb_connection.py` - Wandb连接测试
- `test_wandb_config.py` - Wandb配置测试
- `test_wandb_display.py` - Wandb显示测试
- `test_wandb_step_fix.py` - Wandb步长修复测试
- `test_wandb_training.py` - Wandb训练集成测试
- `test_wandb_validation_upload.py` - Wandb验证数据上传测试

## 使用方法

### 运行单个测试
```bash
# 从项目根目录运行
cd /usr/stud/dira/GraspInClutter/targo
python tests/test_dataset_error_handling.py
```

### 运行所有测试
```bash
# 依次运行所有测试文件
for test_file in tests/test_*.py; do
    echo "运行测试: $test_file"
    python "$test_file"
    echo "完成: $test_file"
    echo "================================"
done
```

## 环境要求

确保已激活正确的conda环境：
- 对于原始TARGO模型：`conda activate targo`
- 对于PointTransformerV3模型：`conda activate ptv3`

## 测试文件开发规范

### 路径导入模板
测试文件应使用以下导入模板：
```python
import sys
from pathlib import Path

# 添加项目路径 - 从tests目录开始
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
```

### 命名规范
- 测试文件命名：`test_<功能名称>.py`
- 测试函数命名：`test_<具体功能>()`或`<功能描述>()`
- 主函数：每个测试文件应有`main()`函数

### 错误处理
- 使用try-except捕获导入错误
- 提供清晰的错误信息和调试输出
- 包含详细的测试结果报告

## 维护说明

### 添加新测试
1. 在tests目录创建新的测试文件
2. 遵循命名规范和导入模板
3. 在此README中添加文件描述
4. 确保测试可以独立运行

### 测试文件清理
- 定期检查测试文件的有效性
- 删除过时的或重复的测试
- 更新文档说明

### 集成测试
考虑创建综合测试脚本，能够：
- 自动运行所有相关测试
- 生成测试报告
- 检查测试覆盖率

## 故障排除

### 常见问题
1. **导入错误**：确保路径设置正确，项目根目录在sys.path中
2. **环境错误**：确认已激活正确的conda环境
3. **依赖缺失**：检查所需的Python包是否已安装
4. **权限问题**：确保对输出文件夹有写入权限

### 调试建议
- 使用`print()`语句输出中间结果
- 检查错误日志文件
- 使用`import pdb; pdb.set_trace()`进行断点调试
- 分步骤运行测试，确定问题位置

## 相关文档
- [数据集错误处理说明](../scripts/README_dataset_error_handling.md)
- [项目README](../PROJECT_README.md)
- [代码规范](.cursorrules) 