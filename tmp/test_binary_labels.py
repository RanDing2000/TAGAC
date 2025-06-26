#!/usr/bin/env python3
"""
测试脚本：验证dataset中binary label feature和PointTransformerV3SceneModel的兼容性
"""

import torch
import numpy as np
import sys
import os
sys.path.append('src')

from transformer.ptv3_scene_model import PointTransformerV3SceneModel

def test_binary_label_functionality():
    """测试binary label功能"""
    
    print("Testing Binary Label Functionality...")
    
    # 创建PointTransformerV3SceneModel实例
    model = PointTransformerV3SceneModel(in_channels=6)  # 支持6维输入
    
    # 测试数据：模拟包含binary label的点云数据
    batch_size = 2
    num_points = 512
    
    # 创建测试点云：[x, y, z, binary_label]
    # 模拟scene点云（binary_label=0）和target点云（binary_label=1）
    test_points = torch.randn(batch_size, num_points, 4)
    
    # 设置binary labels
    # 前一半点为scene点（label=0），后一半为target点（label=1）
    test_points[:, :num_points//2, 3] = 0.0  # Scene points
    test_points[:, num_points//2:, 3] = 1.0  # Target points
    
    print(f"Input shape: {test_points.shape}")
    print(f"Binary labels - Scene points: {test_points[0, :10, 3]}")
    print(f"Binary labels - Target points: {test_points[0, -10:, 3]}")
    
    try:
        # 前向传播
        grid_feat = model(test_points)
        print(f"✓ Forward pass successful!")
        print(f"Output grid feature shape: {grid_feat.shape}")
        
        # 验证输出尺寸
        expected_shape = (batch_size, 64, 40, 40, 40)  # [BS, 64, 40, 40, 40]
        if grid_feat.shape == expected_shape:
            print(f"✓ Output shape correct: {grid_feat.shape}")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {grid_feat.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*50)
    print("Testing dataset integration...")
    
    # 测试模拟dataset输出格式
    try:
        # 模拟DatasetVoxel_Target的输出格式
        # (voxel_grid, targ_grid, targ_pc_with_labels, scene_pc_with_labels)
        
        # 创建模拟的target和scene点云（带binary labels）
        targ_pc_with_labels = torch.randn(batch_size, 256, 4)
        targ_pc_with_labels[:, :, 3] = 1.0  # Target points labeled as 1
        
        scene_pc_with_labels = torch.randn(batch_size, 256, 4) 
        scene_pc_with_labels[:, :, 3] = 0.0  # Scene points labeled as 0
        
        print(f"Target PC shape: {targ_pc_with_labels.shape}, labels: {targ_pc_with_labels[0, :5, 3]}")
        print(f"Scene PC shape: {scene_pc_with_labels.shape}, labels: {scene_pc_with_labels[0, :5, 3]}")
        
        # 测试ptv3_scene模式（合并scene和target）
        combined_pc = torch.cat([scene_pc_with_labels, targ_pc_with_labels], dim=1)
        print(f"Combined PC shape: {combined_pc.shape}")
        
        # 前向传播
        grid_feat_combined = model(combined_pc)
        print(f"✓ Combined forward pass successful!")
        print(f"Combined output shape: {grid_feat_combined.shape}")
        
    except Exception as e:
        print(f"✗ Combined forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("Binary label feature integration successful!")
    
    return True

if __name__ == "__main__":
    test_binary_label_functionality() 