#!/usr/bin/env python3
"""
修复maniskill YCB URDF文件中的mesh路径
为所有filename添加完整路径前缀
"""

import os
import re
from pathlib import Path

def fix_urdf_paths(urdf_dir):
    """
    遍历URDF文件并修复mesh路径
    
    Args:
        urdf_dir: URDF文件目录路径
    """
    urdf_dir = Path(urdf_dir)
    base_path = "/home/ran.ding/projects/TARGO/data/maniskill_ycb/mani_skill2_ycb/collisions"
    
    # 获取所有URDF文件
    urdf_files = list(urdf_dir.glob("*.urdf"))
    
    if not urdf_files:
        print(f"在目录 {urdf_dir} 中没有找到URDF文件")
        return
    
    print(f"找到 {len(urdf_files)} 个URDF文件")
    
    fixed_count = 0
    total_mesh_count = 0
    
    for urdf_file in urdf_files:
        print(f"处理文件: {urdf_file.name}")
        
        # 读取文件内容
        with open(urdf_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有mesh filename
        mesh_pattern = r'<mesh filename="([^"]+)" />'
        matches = re.findall(mesh_pattern, content)
        
        if not matches:
            print(f"  警告: {urdf_file.name} 中没有找到mesh标签")
            continue
        
        print(f"  找到 {len(matches)} 个mesh标签")
        total_mesh_count += len(matches)
        
        modified = False
        
        for match in matches:
            # 检查是否已经有完整路径
            if match.startswith('/'):
                print(f"    跳过已有完整路径: {match}")
                continue
            
            # 检查是否已经有相对路径前缀
            if match.startswith('data/'):
                print(f"    跳过已有相对路径: {match}")
                continue
            
            # 构建完整路径
            full_path = f"{base_path}/{match}"
            
            # 检查文件是否存在
            if os.path.exists(full_path):
                # 替换路径
                content = content.replace(f'filename="{match}"', f'filename="{full_path}"')
                print(f"    修复: {match} -> {full_path}")
                modified = True
            else:
                print(f"    警告: 文件不存在 {full_path}")
        
        # 如果有修改，写回文件
        if modified:
            with open(urdf_file, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"  ✓ 已修复 {urdf_file.name}")
        else:
            print(f"  - 无需修复 {urdf_file.name}")
    
    print(f"\n修复完成!")
    print(f"处理文件数: {len(urdf_files)}")
    print(f"修复文件数: {fixed_count}")
    print(f"总mesh标签数: {total_mesh_count}")

def main():
    """主函数"""
    urdf_dir = "data/maniskill_ycb/mani_skill2_ycb/collisions"
    
    if not os.path.exists(urdf_dir):
        print(f"错误: 目录不存在 {urdf_dir}")
        return
    
    print("开始修复maniskill YCB URDF文件路径...")
    print(f"目标目录: {urdf_dir}")
    print(f"基础路径: /home/ran.ding/projects/TARGO/data/maniskill_ycb/mani_skill2_ycb/collisions")
    print("-" * 80)
    
    fix_urdf_paths(urdf_dir)

if __name__ == "__main__":
    main()
