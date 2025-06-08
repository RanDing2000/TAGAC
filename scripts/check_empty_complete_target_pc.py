#!/usr/bin/env python3
"""
检查 complete_target_pc 是否存在空点云的情况
"""
# python scripts/check_empty_complete_target_pc.py --dataset_root /storage/user/dira/nips_data_version6/combined/targo_dataset --max_scenes 10000

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def check_pc_data(scene_path):
    """
    检查场景文件中的 complete_target_pc 数据
    
    Args:
        scene_path: 场景文件路径
        
    Returns:
        dict: 检查结果
    """
    result = {
        'scene_file': scene_path.name,
        'exists': scene_path.exists(),
        'readable': False,
        'has_complete_target_pc': False,
        'pc_shape': None,
        'pc_points_count': 0,
        'is_empty': False,
        'error': None,
        'available_keys': []
    }
    
    if not result['exists']:
        return result
    
    try:
        # 读取文件
        data = np.load(scene_path, allow_pickle=True)
        result['readable'] = True
        result['available_keys'] = list(data.keys())
        
        # 检查 complete_target_pc
        if 'complete_target_pc' in data:
            result['has_complete_target_pc'] = True
            pc_data = data['complete_target_pc']
            result['pc_shape'] = pc_data.shape
            
            # 检查是否为空
            if pc_data.size == 0:
                result['is_empty'] = True
                result['pc_points_count'] = 0
            elif len(pc_data.shape) >= 1:
                result['pc_points_count'] = pc_data.shape[0]
                result['is_empty'] = (pc_data.shape[0] == 0)
            else:
                result['is_empty'] = True
                result['pc_points_count'] = 0
                
        data.close()
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_pc_data(dataset_root, output_file=None, max_scenes=0):
    """
    分析数据集中 complete_target_pc 的情况
    
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
    
    # 统计信息
    stats = {
        'total_scenes': len(scene_files),
        'readable_scenes': 0,
        'has_complete_pc': 0,
        'empty_pc': 0,
        'valid_pc': 0,
        'unreadable': 0,
        'missing_pc': 0
    }
    
    # 按场景类型分类统计
    scene_types = defaultdict(lambda: {
        'total': 0, 'has_pc': 0, 'empty_pc': 0, 'valid_pc': 0
    })
    
    # 详细结果存储
    results = {
        'empty_pc_scenes': [],
        'missing_pc_scenes': [],
        'unreadable_scenes': [],
        'valid_pc_scenes': []
    }
    
    # 点云大小统计
    pc_sizes = []
    
    # 检查每个场景文件
    for scene_file in tqdm(scene_files, desc="检查 complete_target_pc"):
        result = check_pc_data(scene_file)
        
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
        
        if not result['readable']:
            stats['unreadable'] += 1
            results['unreadable_scenes'].append((scene_name, result['error']))
        elif not result['has_complete_target_pc']:
            stats['missing_pc'] += 1
            results['missing_pc_scenes'].append(scene_name)
        else:
            stats['readable_scenes'] += 1
            stats['has_complete_pc'] += 1
            scene_types[scene_type]['has_pc'] += 1
            
            if result['is_empty']:
                stats['empty_pc'] += 1
                scene_types[scene_type]['empty_pc'] += 1
                results['empty_pc_scenes'].append((scene_name, result['pc_shape'], result['pc_points_count']))
                print(f"发现空点云: {scene_name}, shape: {result['pc_shape']}, points: {result['pc_points_count']}")
            else:
                stats['valid_pc'] += 1
                scene_types[scene_type]['valid_pc'] += 1
                results['valid_pc_scenes'].append((scene_name, result['pc_shape'], result['pc_points_count']))
                pc_sizes.append(result['pc_points_count'])
    
    # 打印统计结果
    print("\n" + "="*60)
    print("complete_target_pc 数据分析报告")
    print("="*60)
    
    print(f"总场景数: {stats['total_scenes']}")
    print(f"可读场景数: {stats['readable_scenes']}")
    print(f"包含 complete_target_pc: {stats['has_complete_pc']}")
    print(f"有效点云: {stats['valid_pc']}")
    print(f"空点云: {stats['empty_pc']}")
    print(f"缺少 complete_target_pc: {stats['missing_pc']}")
    print(f"不可读文件: {stats['unreadable']}")
    
    print("\n按场景类型统计:")
    for scene_type, data in scene_types.items():
        if data['total'] > 0:
            print(f"  {scene_type}:")
            print(f"    总数: {data['total']}")
            print(f"    有点云: {data['has_pc']}")
            print(f"    有效点云: {data['valid_pc']}")
            print(f"    空点云: {data['empty_pc']}")
            if data['has_pc'] > 0:
                empty_rate = data['empty_pc'] / data['has_pc'] * 100
                print(f"    空点云率: {empty_rate:.1f}%")
    
    # 点云大小统计
    if pc_sizes:
        print(f"\n点云大小统计:")
        print(f"  最小点数: {min(pc_sizes)}")
        print(f"  最大点数: {max(pc_sizes)}")
        print(f"  平均点数: {np.mean(pc_sizes):.1f}")
        print(f"  中位数点数: {np.median(pc_sizes):.1f}")
    
    # 显示空点云详情
    if results['empty_pc_scenes']:
        print(f"\n空点云场景详情 ({len(results['empty_pc_scenes'])} 个):")
        print("-" * 50)
        for scene_name, shape, count in results['empty_pc_scenes']:
            print(f"  {scene_name}: shape={shape}, points={count}")
    
    # 显示前10个有效点云的详情
    if results['valid_pc_scenes']:
        print(f"\n前10个有效点云场景详情:")
        print("-" * 50)
        for i, (scene_name, shape, count) in enumerate(results['valid_pc_scenes'][:10]):
            print(f"  {scene_name}: shape={shape}, points={count}")
    
    # 保存详细报告
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("complete_target_pc 详细分析报告\n")
            f.write("="*50 + "\n\n")
            
            f.write("统计摘要:\n")
            f.write(f"总场景数: {stats['total_scenes']}\n")
            f.write(f"可读场景数: {stats['readable_scenes']}\n")
            f.write(f"包含 complete_target_pc: {stats['has_complete_pc']}\n")
            f.write(f"有效点云: {stats['valid_pc']}\n")
            f.write(f"空点云: {stats['empty_pc']}\n")
            f.write(f"缺少 complete_target_pc: {stats['missing_pc']}\n")
            f.write(f"不可读文件: {stats['unreadable']}\n\n")
            
            # 空点云详情
            if results['empty_pc_scenes']:
                f.write(f"空点云场景列表 ({len(results['empty_pc_scenes'])} 个):\n")
                f.write("-" * 30 + "\n")
                for scene_name, shape, count in results['empty_pc_scenes']:
                    f.write(f"  {scene_name}: shape={shape}, points={count}\n")
                f.write("\n")
            
            # 场景类型统计
            f.write("按场景类型详细统计:\n")
            f.write("-" * 30 + "\n")
            for scene_type, data in scene_types.items():
                if data['total'] > 0:
                    f.write(f"{scene_type}:\n")
                    f.write(f"  总数: {data['total']}\n")
                    f.write(f"  有点云: {data['has_pc']}\n")
                    f.write(f"  有效点云: {data['valid_pc']}\n")
                    f.write(f"  空点云: {data['empty_pc']}\n")
                    if data['has_pc'] > 0:
                        empty_rate = data['empty_pc'] / data['has_pc'] * 100
                        f.write(f"  空点云率: {empty_rate:.1f}%\n")
                    f.write("\n")
        
        print(f"\n详细报告已保存到: {output_path}")
    
    return stats, results


def check_specific_scenes_pc(dataset_root, scene_ids):
    """
    检查特定场景的 complete_target_pc 情况
    
    Args:
        dataset_root: 数据集根目录
        scene_ids: 场景ID列表
    """
    dataset_path = Path(dataset_root)
    scenes_dir = dataset_path / "scenes"
    
    print(f"检查 {len(scene_ids)} 个特定场景的 complete_target_pc...")
    
    for scene_id in scene_ids:
        scene_file = scenes_dir / f"{scene_id}.npz"
        result = check_pc_data(scene_file)
        
        print(f"\n{scene_id}:")
        print(f"  存在: {result['exists']}")
        print(f"  可读: {result['readable']}")
        print(f"  有 complete_target_pc: {result['has_complete_target_pc']}")
        
        if result['has_complete_target_pc']:
            print(f"  点云形状: {result['pc_shape']}")
            print(f"  点数: {result['pc_points_count']}")
            print(f"  是否为空: {result['is_empty']}")
        
        if result['error']:
            print(f"  错误: {result['error']}")
        
        print(f"  可用键: {result['available_keys']}")


def main():
    parser = argparse.ArgumentParser(description="检查 complete_target_pc 中的空点云情况")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="数据集根目录路径")
    parser.add_argument("--output_file", type=str, default="complete_target_pc_analysis.txt",
                        help="输出报告文件路径")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="最大检查场景数 (0表示检查所有)")
    parser.add_argument("--check_scenes", type=str, nargs="+",
                        help="检查特定场景ID列表")
    
    args = parser.parse_args()
    
    if args.check_scenes:
        # 检查特定场景
        check_specific_scenes_pc(args.dataset_root, args.check_scenes)
    else:
        # 分析整个数据集
        analyze_pc_data(args.dataset_root, args.output_file, args.max_scenes)


if __name__ == "__main__":
    main() 