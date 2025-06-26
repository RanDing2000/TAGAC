#!/usr/bin/env python3
"""
检查哪些场景文件缺少 complete_target_pc 和 complete_target_tsdf 数据
"""

# python -m pdb scripts/check_missing_complete_target.py --dataset_root /storage/user/dira/nips_data_version6/combined/targo_dataset --max_scenes 100


import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict


def check_scene_file(scene_path, required_fields=None):
    """
    检查场景文件是否包含必需的字段
    
    Args:
        scene_path: 场景文件路径
        required_fields: 需要检查的字段列表
        
    Returns:
        dict: 检查结果
    """
    if required_fields is None:
        required_fields = ['complete_target_pc', 'complete_target_tsdf']
    
    result = {
        'scene_file': scene_path.name,
        'exists': scene_path.exists(),
        'readable': False,
        'available_keys': [],
        'missing_fields': [],
        'has_complete_data': False,
        'error': None
    }
    
    if not result['exists']:
        result['missing_fields'] = required_fields
        return result
    
    try:
        # 尝试读取文件
        data = np.load(scene_path, allow_pickle=True)
        result['readable'] = True
        result['available_keys'] = list(data.keys())
        
        # 检查必需字段
        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            else:
                # 检查数据是否有效
                field_data = data[field]
                if field == 'complete_target_pc':
                    if field_data.size == 0 or field_data.shape[0] == 0:
                        missing_fields.append(f"{field}_empty")
                elif field == 'complete_target_tsdf':
                    if field_data.size == 0:
                        missing_fields.append(f"{field}_empty")
        
        result['missing_fields'] = missing_fields
        result['has_complete_data'] = len(missing_fields) == 0
        
    except Exception as e:
        result['error'] = str(e)
        result['missing_fields'] = required_fields
    
    return result


def analyze_dataset(dataset_root, output_file=None, max_scenes=0):
    """
    分析整个数据集的完整性
    
    Args:
        dataset_root: 数据集根目录
        output_file: 输出报告文件路径
        max_scenes: 最大检查场景数 (0表示检查所有)
    """
    dataset_path = Path(dataset_root)
    scenes_dir = dataset_path / "scenes"
    
    if not scenes_dir.exists():
        print(f"错误: 场景目录不存在: {scenes_dir}")
        return
    
    # 获取所有场景文件
    scene_files = list(scenes_dir.glob("*.npz"))
    if max_scenes > 0:
        scene_files = scene_files[:max_scenes]
    
    print(f"找到 {len(scene_files)} 个场景文件待检查")
    
    # 分析结果统计
    stats = {
        'total_scenes': len(scene_files),
        'complete_scenes': 0,
        'missing_pc': 0,
        'missing_tsdf': 0,
        'missing_both': 0,
        'unreadable': 0,
        'not_exist': 0,
        'empty_data': 0
    }
    
    # 存储详细问题
    problems = {
        'missing_pc': [],
        'missing_tsdf': [],
        'missing_both': [],
        'unreadable': [],
        'not_exist': [],
        'empty_data': []
    }
    
    # 按场景类型分类
    scene_types = defaultdict(lambda: {'total': 0, 'complete': 0, 'missing': 0})
    
    # 检查每个场景文件
    for scene_file in tqdm(scene_files, desc="检查场景文件"):
        result = check_scene_file(scene_file)
        
        # 确定场景类型
        scene_name = scene_file.stem
        if '_c_' in scene_name:
            scene_type = 'cluttered'
        elif '_s_' in scene_name:
            scene_type = 'single'
        elif '_d_' in scene_name:
            scene_type = 'double'
        else:
            scene_type = 'unknown'
        
        scene_types[scene_type]['total'] += 1
        
        # 统计结果
        if not result['exists']:
            stats['not_exist'] += 1
            problems['not_exist'].append(scene_name)
            scene_types[scene_type]['missing'] += 1
        elif not result['readable']:
            stats['unreadable'] += 1
            problems['unreadable'].append((scene_name, result['error']))
            scene_types[scene_type]['missing'] += 1
        elif result['has_complete_data']:
            stats['complete_scenes'] += 1
            scene_types[scene_type]['complete'] += 1
        else:
            # 分析缺失类型
            missing = result['missing_fields']
            scene_types[scene_type]['missing'] += 1
            
            has_pc_issue = any('complete_target_pc' in field for field in missing)
            has_tsdf_issue = any('complete_target_tsdf' in field for field in missing)
            has_empty_data = any('_empty' in field for field in missing)
            
            if has_empty_data:
                stats['empty_data'] += 1
                problems['empty_data'].append((scene_name, missing))
            
            if has_pc_issue and has_tsdf_issue:
                stats['missing_both'] += 1
                problems['missing_both'].append((scene_name, missing))
            elif has_pc_issue:
                stats['missing_pc'] += 1
                problems['missing_pc'].append((scene_name, missing))
            elif has_tsdf_issue:
                stats['missing_tsdf'] += 1
                problems['missing_tsdf'].append((scene_name, missing))
    
    # 打印统计结果
    print("\n" + "="*60)
    print("数据集完整性分析报告")
    print("="*60)
    
    print(f"总场景数: {stats['total_scenes']}")
    print(f"完整场景数: {stats['complete_scenes']} ({stats['complete_scenes']/stats['total_scenes']*100:.1f}%)")
    print(f"缺少 complete_target_pc: {stats['missing_pc']}")
    print(f"缺少 complete_target_tsdf: {stats['missing_tsdf']}")
    print(f"两者都缺少: {stats['missing_both']}")
    print(f"空数据: {stats['empty_data']}")
    print(f"不可读文件: {stats['unreadable']}")
    print(f"不存在文件: {stats['not_exist']}")
    
    print("\n按场景类型统计:")
    for scene_type, data in scene_types.items():
        if data['total'] > 0:
            complete_rate = data['complete'] / data['total'] * 100
            print(f"  {scene_type}: {data['complete']}/{data['total']} 完整 ({complete_rate:.1f}%)")
    
    # 详细问题报告
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("数据集完整性详细报告\n")
            f.write("="*50 + "\n\n")
            
            f.write("统计摘要:\n")
            f.write(f"总场景数: {stats['total_scenes']}\n")
            f.write(f"完整场景数: {stats['complete_scenes']} ({stats['complete_scenes']/stats['total_scenes']*100:.1f}%)\n")
            f.write(f"缺少 complete_target_pc: {stats['missing_pc']}\n")
            f.write(f"缺少 complete_target_tsdf: {stats['missing_tsdf']}\n")
            f.write(f"两者都缺少: {stats['missing_both']}\n")
            f.write(f"空数据: {stats['empty_data']}\n")
            f.write(f"不可读文件: {stats['unreadable']}\n")
            f.write(f"不存在文件: {stats['not_exist']}\n\n")
            
            # 详细问题列表
            for problem_type, problem_list in problems.items():
                if problem_list:
                    f.write(f"\n{problem_type.upper()} ({len(problem_list)} 个):\n")
                    f.write("-" * 30 + "\n")
                    for item in problem_list:
                        if isinstance(item, tuple):
                            f.write(f"  {item[0]}: {item[1]}\n")
                        else:
                            f.write(f"  {item}\n")
        
        print(f"\n详细报告已保存到: {output_path}")
    
    # 显示前10个有问题的文件
    print("\n前10个有问题的场景:")
    print("-" * 30)
    problem_count = 0
    for problem_type, problem_list in problems.items():
        if problem_count >= 10:
            break
        for item in problem_list[:10-problem_count]:
            if isinstance(item, tuple):
                print(f"  {item[0]} ({problem_type}): {item[1]}")
            else:
                print(f"  {item} ({problem_type})")
            problem_count += 1
            if problem_count >= 10:
                break
    
    return stats, problems


def check_specific_scenes(dataset_root, scene_ids):
    """
    检查特定场景的完整性
    
    Args:
        dataset_root: 数据集根目录
        scene_ids: 场景ID列表
    """
    dataset_path = Path(dataset_root)
    scenes_dir = dataset_path / "scenes"
    
    print(f"检查 {len(scene_ids)} 个特定场景...")
    
    for scene_id in scene_ids:
        scene_file = scenes_dir / f"{scene_id}.npz"
        result = check_scene_file(scene_file)
        
        status = "✓ 完整" if result['has_complete_data'] else "✗ 不完整"
        print(f"{scene_id}: {status}")
        
        if not result['has_complete_data']:
            if not result['exists']:
                print(f"  -> 文件不存在")
            elif not result['readable']:
                print(f"  -> 读取错误: {result['error']}")
            else:
                print(f"  -> 缺少字段: {result['missing_fields']}")
                print(f"  -> 可用字段: {result['available_keys']}")


def main():
    parser = argparse.ArgumentParser(description="检查数据集中缺少完整目标数据的场景")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="数据集根目录路径")
    parser.add_argument("--output_file", type=str, default="missing_complete_target_report.txt",
                        help="输出报告文件路径")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="最大检查场景数 (0表示检查所有)")
    parser.add_argument("--check_scenes", type=str, nargs="+",
                        help="检查特定场景ID列表")
    
    args = parser.parse_args()
    
    if args.check_scenes:
        # 检查特定场景
        check_specific_scenes(args.dataset_root, args.check_scenes)
    else:
        # 分析整个数据集
        analyze_dataset(args.dataset_root, args.output_file, args.max_scenes)


if __name__ == "__main__":
    main() 