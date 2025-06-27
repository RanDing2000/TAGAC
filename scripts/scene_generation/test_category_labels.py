#!/usr/bin/env python3
"""
测试脚本：验证类别标签和CLIP特征添加功能
"""

import os
import numpy as np
import json
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_enhanced_npz_file(npz_path, category_mapping_path):
    """
    测试增强的npz文件，验证新增字段的正确性。
    
    Args:
        npz_path: 增强的npz文件路径
        category_mapping_path: 类别映射文件路径
    """
    print(f"Testing file: {npz_path}")
    
    # Load the enhanced npz file
    data = np.load(npz_path, allow_pickle=True)
    
    # Load category mapping
    with open(category_mapping_path, 'r') as f:
        category_mapping = json.load(f)
    
    # Check required fields
    required_fields = ['points', 'instance_labels', 'category_labels', 'object_labels', 'clip_features']
    for field in required_fields:
        if field not in data:
            print(f"ERROR: Missing required field '{field}'")
            return False
        print(f"✓ Found field '{field}' with shape {data[field].shape}")
    
    # Check data consistency
    N = len(data['points'])
    print(f"✓ Total points: {N}")
    
    # Check instance labels
    instance_labels = data['instance_labels']
    scene_points = np.sum(instance_labels == 0)
    target_points = np.sum(instance_labels == 1)
    print(f"✓ Scene points: {scene_points}, Target points: {target_points}")
    
    # Check category labels
    category_labels = data['category_labels']
    unique_categories = np.unique(category_labels)
    print(f"✓ Unique categories: {len(unique_categories)}")
    for cat_idx in unique_categories:
        cat_name = category_mapping.get(str(cat_idx), f"unknown_{cat_idx}")
        count = np.sum(category_labels == cat_idx)
        print(f"  - Category {cat_idx} ({cat_name}): {count} points")
    
    # Check object labels
    object_labels = data['object_labels']
    unique_objects = np.unique(object_labels)
    print(f"✓ Unique objects: {len(unique_objects)}")
    
    # Check CLIP features
    clip_features = data['clip_features']
    print(f"✓ CLIP features shape: {clip_features.shape}")
    print(f"✓ CLIP features range: [{clip_features.min():.4f}, {clip_features.max():.4f}]")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(clip_features)):
        print("WARNING: Found NaN values in CLIP features")
    if np.any(np.isinf(clip_features)):
        print("WARNING: Found infinite values in CLIP features")
    
    print("✓ All tests passed!")
    return True

def test_category_mapping_files(output_dir):
    """
    测试类别映射文件。
    
    Args:
        output_dir: 输出目录路径
    """
    print(f"\nTesting category mapping files in: {output_dir}")
    
    # Check category_mapping.json
    category_mapping_path = output_dir / "category_mapping.json"
    if category_mapping_path.exists():
        with open(category_mapping_path, 'r') as f:
            category_mapping = json.load(f)
        print(f"✓ Found category_mapping.json with {len(category_mapping)} categories")
        print("  Categories:", list(category_mapping.values())[:10], "...")
    else:
        print("ERROR: category_mapping.json not found")
        return False
    
    # Check object_category_mapping.json
    object_mapping_path = output_dir / "object_category_mapping.json"
    if object_mapping_path.exists():
        with open(object_mapping_path, 'r') as f:
            object_mapping = json.load(f)
        print(f"✓ Found object_category_mapping.json with {len(object_mapping)} object mappings")
    else:
        print("ERROR: object_category_mapping.json not found")
        return False
    
    return True

def main():
    """
    主测试函数。
    """
    # Test parameters
    output_dir = Path("/usr/stud/dira/GraspInClutter/targo/data_scenes/targo_dataset/scenes_enhanced")
    
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        print("Please run add_category_labels_clip_feature.py first")
        return
    
    print("=" * 60)
    print("TESTING CATEGORY LABELS AND CLIP FEATURES")
    print("=" * 60)
    
    # Test category mapping files
    if not test_category_mapping_files(output_dir):
        return
    
    # Find enhanced npz files
    npz_files = list(output_dir.glob("*.npz"))
    if not npz_files:
        print("ERROR: No enhanced npz files found")
        return
    
    print(f"\nFound {len(npz_files)} enhanced npz files")
    
    # Test a few files
    test_files = npz_files[:3]  # Test first 3 files
    category_mapping_path = output_dir / "category_mapping.json"
    
    success_count = 0
    for npz_file in test_files:
        print(f"\n{'='*40}")
        if test_enhanced_npz_file(npz_file, category_mapping_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {success_count}/{len(test_files)} files passed")
    
    if success_count == len(test_files):
        print("✓ All tests passed! The enhancement process is working correctly.")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 