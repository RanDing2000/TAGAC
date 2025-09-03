#!/usr/bin/env python3
import os
from pathlib import Path

# 获取mess_kitchen文件夹下的所有目录名
mess_kitchen_path = Path("data/gso/google_scanned_objects/mess_kitchen")
mess_kitchen_dirs = []
if mess_kitchen_path.exists():
    mess_kitchen_dirs = [d.name for d in mess_kitchen_path.iterdir() if d.is_dir()]

print(f"mess_kitchen文件夹下有 {len(mess_kitchen_dirs)} 个目录:")
for i, dir_name in enumerate(mess_kitchen_dirs[:10]):  # 只显示前10个
    print(f"  {i+1}. {dir_name}")
if len(mess_kitchen_dirs) > 10:
    print(f"  ... 还有 {len(mess_kitchen_dirs) - 10} 个目录")

# 查找所有可能的txt文件
print(f"\n查找所有txt文件:")
all_txt_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith('.txt'):
            all_txt_files.append(os.path.join(root, file))

print(f"找到 {len(all_txt_files)} 个txt文件:")
for txt_file in all_txt_files[:20]:  # 只显示前20个
    print(f"  {txt_file}")
if len(all_txt_files) > 20:
    print(f"  ... 还有 {len(all_txt_files) - 20} 个txt文件")

# 查找包含"kitchen"或"mess"的txt文件
kitchen_txt_files = []
for txt_file in all_txt_files:
    if "kitchen" in txt_file.lower() or "mess" in txt_file.lower():
        kitchen_txt_files.append(txt_file)

print(f"\n包含'kitchen'或'mess'的txt文件:")
for txt_file in kitchen_txt_files:
    print(f"  {txt_file}")

# 检查每个txt文件的内容，看是否包含mess_kitchen中的目录名
print(f"\n检查txt文件内容:")
for txt_file in kitchen_txt_files:
    try:
        with open(txt_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            
            # 检查是否包含mess_kitchen中的目录名
            matches = []
            for line in lines:
                line = line.strip()
                if line in mess_kitchen_dirs:
                    matches.append(line)
            
            if matches:
                print(f"\n文件 {txt_file} 包含以下mess_kitchen目录名:")
                for match in matches[:10]:  # 只显示前10个
                    print(f"  ✓ {match}")
                if len(matches) > 10:
                    print(f"  ... 还有 {len(matches) - 10} 个匹配")
                print(f"  总共 {len(matches)} 个匹配")
    except Exception as e:
        print(f"读取文件 {txt_file} 时出错: {e}")

# 如果没有找到相关文件，显示mess_kitchen中的所有目录名
if not kitchen_txt_files:
    print(f"\n没有找到相关的txt文件。mess_kitchen中的所有目录名:")
    for i, dir_name in enumerate(mess_kitchen_dirs):
        print(f"  {i+1:2d}. {dir_name}")
