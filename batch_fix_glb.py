#!/usr/bin/env python3
"""
批量修复GLB文件中的颜色因子问题
参考fix_glb.py的逻辑，修复g1b_files目录下所有的GLB文件
"""

import os
import numpy as np
import trimesh
import sys
from pathlib import Path
import argparse
import json

def fix_glb_colors(glb_path, output_path=None):
    """修复GLB文件中的颜色因子问题"""
    
    if output_path is None:
        base_name = os.path.splitext(glb_path)[0]
        output_path = f"{base_name}_fixed.glb"
    
    print(f"修复GLB文件: {glb_path}")
    print(f"输出文件: {output_path}")
    
    try:
        # 加载GLB文件
        mesh = trimesh.load(glb_path, process=False)
        
        if isinstance(mesh, trimesh.Scene):
            print(f"场景中的几何体数量: {len(mesh.geometry)}")
            
            for name, geom in mesh.geometry.items():
                print(f"\n处理几何体: {name}")
                
                if hasattr(geom, 'visual') and geom.visual is not None:
                    if hasattr(geom.visual, 'material'):
                        material = geom.visual.material
                        
                        # 检查基础颜色因子
                        if hasattr(material, 'baseColorFactor'):
                            original_factor = material.baseColorFactor
                            print(f"  原始基础颜色因子: {original_factor}")
                            
                            # 如果基础颜色因子是纯白色，将其设置为None或[1,1,1,1]
                            if original_factor is not None and np.all(original_factor == [255, 255, 255, 255]):
                                print(f"  检测到纯白色基础颜色因子，正在修复...")
                                # 方法1: 设置为None（让纹理颜色生效）
                                material.baseColorFactor = None
                                print(f"  修复后基础颜色因子: {material.baseColorFactor}")
                            elif original_factor is not None and np.all(original_factor == [1, 1, 1, 1]):
                                print(f"  检测到单位白色基础颜色因子，正在修复...")
                                material.baseColorFactor = None
                                print(f"  修复后基础颜色因子: {material.baseColorFactor}")
                            else:
                                print(f"  基础颜色因子正常，无需修复")
                        
                        # 检查是否有纹理
                        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                            print(f"  基础颜色纹理: {material.baseColorTexture}")
                            print(f"  纹理模式: {material.baseColorTexture.mode}")
                            print(f"  纹理尺寸: {material.baseColorTexture.size}")
                        else:
                            print(f"  无基础颜色纹理")
        
        # 保存修复后的GLB文件
        mesh.export(output_path)
        print(f"\n修复后的GLB文件已保存: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"修复GLB文件时出错: {e}")
        return None

def batch_fix_glb_files(g1b_dir, backup_original=True):
    """批量修复g1b_files目录下的所有GLB文件"""
    
    g1b_path = Path(g1b_dir)
    if not g1b_path.exists():
        print(f"目录不存在: {g1b_path}")
        return
    
    # 查找所有GLB文件
    glb_files = list(g1b_path.glob("*.glb"))
    print(f"找到 {len(glb_files)} 个GLB文件")
    
    if len(glb_files) == 0:
        print("没有找到GLB文件")
        return
    
    # 创建备份目录
    if backup_original:
        backup_dir = g1b_path / "backup_original"
        backup_dir.mkdir(exist_ok=True)
        print(f"备份目录: {backup_dir}")
    
    # 修复统计
    fixed_count = 0
    failed_count = 0
    fixed_files = []
    failed_files = []
    
    for glb_file in glb_files:
        print(f"\n{'='*60}")
        print(f"处理文件 {fixed_count + failed_count + 1}/{len(glb_files)}: {glb_file.name}")
        print(f"{'='*60}")
        
        try:
            # 备份原始文件
            if backup_original:
                backup_file = backup_dir / glb_file.name
                import shutil
                shutil.copy2(glb_file, backup_file)
                print(f"已备份原始文件到: {backup_file}")
            
            # 修复GLB文件
            fixed_path = fix_glb_colors(str(glb_file))
            
            if fixed_path and os.path.exists(fixed_path):
                # 替换原文件
                import shutil
                shutil.move(fixed_path, str(glb_file))
                print(f"已用修复后的文件替换原文件: {glb_file}")
                fixed_count += 1
                fixed_files.append(glb_file.name)
            else:
                print(f"修复失败: {glb_file.name}")
                failed_count += 1
                failed_files.append(glb_file.name)
                
        except Exception as e:
            print(f"处理文件时出错: {e}")
            failed_count += 1
            failed_files.append(glb_file.name)
    
    # 输出统计结果
    print(f"\n{'='*60}")
    print("批量修复完成")
    print(f"{'='*60}")
    print(f"总文件数: {len(glb_files)}")
    print(f"修复成功: {fixed_count}")
    print(f"修复失败: {failed_count}")
    
    if fixed_files:
        print(f"\n修复成功的文件:")
        for file in fixed_files:
            print(f"  ✓ {file}")
    
    if failed_files:
        print(f"\n修复失败的文件:")
        for file in failed_files:
            print(f"  ✗ {file}")
    
    # 保存修复报告
    report = {
        "total_files": len(glb_files),
        "fixed_count": fixed_count,
        "failed_count": failed_count,
        "fixed_files": fixed_files,
        "failed_files": failed_files,
        "backup_created": backup_original
    }
    
    report_path = g1b_path / "fix_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n修复报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="批量修复GLB文件中的颜色因子问题")
    parser.add_argument("--g1b-dir", type=str, 
                       default="/home/ran.ding/projects/TARGO/messy_kitchen_scenes/gso_pile_scenes/g1b_files",
                       help="g1b_files目录路径")
    parser.add_argument("--no-backup", action="store_true", 
                       help="不备份原始文件")
    
    args = parser.parse_args()
    
    print("批量修复GLB文件")
    print(f"目标目录: {args.g1b_dir}")
    print(f"备份原始文件: {not args.no_backup}")
    
    batch_fix_glb_files(args.g1b_dir, backup_original=not args.no_backup)

if __name__ == "__main__":
    main()
