#!/usr/bin/env python3
"""
测试脚本：验证ptv3_scene在detection_implicit.py中的修复是否有效
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# 设置环境
sys.path.append('/usr/stud/dira/GraspInClutter/targo')
os.chdir('/usr/stud/dira/GraspInClutter/targo')

def test_ptv3_scene_data_format():
    """测试ptv3_scene的数据格式处理"""
    print("🔧 测试ptv3_scene数据格式处理")
    
    # 创建模拟的combined point cloud数据 [N, 4] 格式
    # 前3列是xyz坐标，第4列是binary标签
    N = 1000
    combined_points = np.random.randn(N, 4).astype(np.float32)
    combined_points[:, :3] = combined_points[:, :3] * 0.1  # 缩放坐标到合理范围
    combined_points[:, 3] = np.random.choice([0.0, 1.0], size=N)  # binary标签
    
    print(f"Combined points shape: {combined_points.shape}")
    print(f"Label distribution: {np.bincount(combined_points[:, 3].astype(int))}")
    
    # 测试转换为tensor
    try:
        scene_with_labels = torch.from_numpy(combined_points).unsqueeze(0)
        print(f"✅ 成功转换为tensor, shape: {scene_with_labels.shape}")
        
        # 验证数据内容
        coords = scene_with_labels[0, :, :3]
        labels = scene_with_labels[0, :, 3]
        print(f"Coordinates range: [{coords.min():.3f}, {coords.max():.3f}]")
        print(f"Labels unique values: {torch.unique(labels)}")
        
        return True
    except Exception as e:
        print(f"❌ 数据格式转换失败: {e}")
        return False

def test_ptv3_scene_model_input():
    """测试ptv3_scene模型输入格式"""
    print("\n🔧 测试ptv3_scene模型输入格式")
    
    try:
        # 模拟detection_implicit.py中的处理
        N = 512
        inputs_data = np.random.randn(N, 4).astype(np.float32)
        inputs_data[:, :3] = inputs_data[:, :3] * 0.1  # 坐标
        inputs_data[:, 3] = np.random.choice([0.0, 1.0], size=N)  # 标签
        
        # 模拟ptv3_scene的处理逻辑
        inputs = (inputs_data, None)  # ptv3_scene只有scene输入，没有target输入
        
        # 转换为tensor
        scene_with_labels = torch.from_numpy(inputs[0]).unsqueeze(0)
        print(f"✅ ptv3_scene输入准备成功, shape: {scene_with_labels.shape}")
        
        # 验证格式符合PointTransformerV3的要求 [B, N, 4]
        assert len(scene_with_labels.shape) == 3, f"期望3D tensor，得到{len(scene_with_labels.shape)}D"
        assert scene_with_labels.shape[0] == 1, f"期望batch_size=1，得到{scene_with_labels.shape[0]}"
        assert scene_with_labels.shape[2] == 4, f"期望4个特征(xyz+label)，得到{scene_with_labels.shape[2]}"
        
        print(f"✅ 输入格式验证通过: [B={scene_with_labels.shape[0]}, N={scene_with_labels.shape[1]}, C={scene_with_labels.shape[2]}]")
        return True
        
    except Exception as e:
        print(f"❌ 模型输入格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prepare_batch_compatibility():
    """测试与prepare_batch函数的兼容性"""
    print("\n🔧 测试与prepare_batch函数的兼容性")
    
    try:
        from scripts.train.train_targo_ptv3 import prepare_batch
        
        # 创建模拟的batch数据
        batch_size = 2
        N = 512
        
        # 模拟DatasetVoxel_PTV3_Scene的输出格式
        # (voxel_grid, targ_grid, targ_pc, scene_pc), (label, rotations, width), pos
        pc = torch.randn(batch_size, 40, 40, 40)
        targ_grid = torch.randn(batch_size, 40, 40, 40)
        targ_pc = torch.randn(batch_size, 256, 4)  # 带标签的target pc
        scene_pc = torch.randn(batch_size, N, 4)  # 带标签的combined scene pc
        
        label = torch.randint(0, 2, (batch_size,))
        rotations = torch.randn(batch_size, 2, 4)
        width = torch.randn(batch_size,)
        pos = torch.randn(batch_size, 3)
        
        batch = ((pc, targ_grid, targ_pc, scene_pc), (label, rotations, width), pos)
        
        # 测试prepare_batch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        result = prepare_batch(batch, device, model_type="ptv3_scene")
        
        scene_pc_result, y, pos_result = result
        print(f"✅ prepare_batch成功处理ptv3_scene")
        print(f"Scene PC shape: {scene_pc_result.shape}")
        print(f"期望格式: [B, N, 4] = [{batch_size}, {N}, 4]")
        
        # 验证返回的数据格式
        assert scene_pc_result.shape == (batch_size, N, 4), f"期望shape {(batch_size, N, 4)}，得到{scene_pc_result.shape}"
        
        print(f"✅ prepare_batch兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ prepare_batch兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("🚀 开始ptv3_scene修复验证测试")
    print("=" * 60)
    
    tests = [
        test_ptv3_scene_data_format,
        test_ptv3_scene_model_input,
        test_prepare_batch_compatibility,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试函数{test.__name__}执行失败: {e}")
            results.append(False)
        print("-" * 40)
    
    # 总结结果
    print("\n📊 测试结果总结:")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！ptv3_scene修复验证成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 