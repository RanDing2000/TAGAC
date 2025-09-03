#!/usr/bin/env python3
"""
测试DatasetVoxel_PTV3_Scene的错误处理机制
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

def test_ptv3_scene_error_handling():
    """测试PTV3 Scene数据集的错误处理"""
    try:
        from src.vgn.dataset_voxel import DatasetVoxel_PTV3_Scene
        
        # 使用测试数据路径
        raw_root = Path("data_scenes/targo_dataset")
        root = raw_root
        
        print("创建DatasetVoxel_PTV3_Scene数据集...")
        dataset = DatasetVoxel_PTV3_Scene(
            root=root,
            raw_root=raw_root,
            ablation_dataset="1_100000",  # 使用很小的数据集进行快速测试
            model_type="ptv3_scene",
            use_complete_targ=True,
            debug=True,
            logdir=Path("./test_output")
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试几个样本
        success_count = 0
        error_count = 0
        
        for i in range(min(5, len(dataset))):
            try:
                print(f"\n测试样本 {i}...")
                x, y, pos = dataset[i]
                print(f"  成功加载样本 {i}")
                print(f"  x类型: {type(x)}, 长度: {len(x) if hasattr(x, '__len__') else 'N/A'}")
                print(f"  y类型: {type(y)}, 长度: {len(y) if hasattr(y, '__len__') else 'N/A'}")
                print(f"  pos形状: {pos.shape if hasattr(pos, 'shape') else 'N/A'}")
                success_count += 1
            except Exception as e:
                print(f"  样本 {i} 加载失败: {e}")
                error_count += 1
        
        print(f"\n测试结果:")
        print(f"  成功加载: {success_count}")
        print(f"  加载失败: {error_count}")
        
        # 检查错误日志文件
        error_log_file = Path("/home/ran.ding/projects/TARGO/data/_check_results/dataset_error_scenes.txt")
        if error_log_file.exists():
            print(f"\n错误日志文件存在: {error_log_file}")
            with open(error_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"错误日志行数: {len(lines)}")
                if lines:
                    print("最新的几条错误记录:")
                    for line in lines[-5:]:  # 显示最后5条
                        print(f"  {line.strip()}")
        else:
            print(f"\n错误日志文件不存在: {error_log_file}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DatasetVoxel_PTV3_Scene 错误处理测试")
    print("=" * 60)
    
    success = test_ptv3_scene_error_handling()
    
    if success:
        print("\n✅ 测试完成")
    else:
        print("\n❌ 测试失败")
        sys.exit(1) 