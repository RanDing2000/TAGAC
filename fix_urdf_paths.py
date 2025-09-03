#!/usr/bin/env python3
"""
修复URDF文件中的错误路径
"""

import os
import re
from pathlib import Path
import glob

def fix_urdf_paths(urdf_dir):
    """修复URDF文件中的mesh路径"""
    
    urdf_dir = Path(urdf_dir)
    if not urdf_dir.exists():
        print(f"URDF目录不存在: {urdf_dir}")
        return
    
    # 查找所有URDF文件
    urdf_files = list(urdf_dir.glob("*.urdf"))
    print(f"找到 {len(urdf_files)} 个URDF文件")
    
    fixed_count = 0
    
    for urdf_file in urdf_files:
        print(f"处理: {urdf_file.name}")
        
        # 读取文件内容
        with open(urdf_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找mesh filename行
        mesh_pattern = r'<mesh filename="([^"]+)" />'
        matches = re.findall(mesh_pattern, content)
        
        modified = False
        for match in matches:
            # 检查是否是相对路径
            if match.startswith('ShapeNetSem-backup/'):
                # 构建正确的绝对路径
                correct_path = f"data/acronym/{match}"
                
                # 检查文件是否存在
                if os.path.exists(correct_path):
                    # 替换路径
                    content = content.replace(f'filename="{match}"', f'filename="{correct_path}"')
                    print(f"  修复路径: {match} -> {correct_path}")
                    modified = True
                else:
                    print(f"  警告: 文件不存在: {correct_path}")
        
        # 写回文件
        if modified:
            with open(urdf_file, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
    
    print(f"修复了 {fixed_count} 个URDF文件")

def check_mesh_files():
    """检查mesh文件是否存在"""
    
    # 检查一些示例文件
    test_files = [
        "data/acronym/ShapeNetSem-backup/models-OBJ/models_center/8ac6b93f8d93c400923ef79fcb120ce8.obj",
        "data/acronym/ShapeNetSem-backup/models-OBJ/models_center/a924eb3037129eaff8095890d92b7d6c.obj"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件不存在: {file_path}")

def main():
    """主函数"""
    
    print("=== 检查mesh文件 ===")
    check_mesh_files()
    
    print("\n=== 修复URDF文件路径 ===")
    urdf_dir = "data/acronym/urdfs_acronym"
    fix_urdf_paths(urdf_dir)

if __name__ == "__main__":
    main()
