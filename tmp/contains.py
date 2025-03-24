import os
import json
import shutil
import re

# 读取category_type.json文件
with open('category_type.json', 'r') as f:
    category_type = json.load(f)

# 源目录和目标目录
src_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/collisions"
dest_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/collisions_tabletop"

# 确保目标目录存在
os.makedirs(dest_dir, exist_ok=True)

# 统计变量
total_files = 0
kept_files = 0
tabletop_categories = set()
other_categories = set()

# 保存根文件名到类别的映射
root_to_category = {}

# 第一遍遍历：建立根文件名到类别的映射
print("第一遍：识别所有文件的类别...")
for filename in os.listdir(src_dir):
    # 提取根文件名和类别
    # 例如：从 "1Shelves_1e3df0ab57e8ca8587f357007f9e75d1_collision.obj" 提取
    # 根文件名: "1Shelves_1e3df0ab57e8ca8587f357007f9e75d1"
    # 类别: "1Shelves"
    match = re.match(r'([^_]+)_(.*?)(?:_collision|_visual|\.urdf)?(?:\.obj)?$', filename)
    if match:
        category = match.group(1)
        root_name = f"{category}_{match.group(2)}"
        root_to_category[root_name] = category
        total_files += 1

# 第二遍遍历：根据类别筛选文件
print("第二遍：筛选并复制tabletop对象文件...")
processed_roots = set()  # 跟踪已处理的根文件名

for filename in os.listdir(src_dir):
    # 提取根文件名和类别
    match = re.match(r'([^_]+)_(.*?)(?:_collision|_visual|\.urdf)?(?:\.obj)?$', filename)
    if not match:
        continue
        
    category = match.group(1)
    root_name = f"{category}_{match.group(2)}"
    
    # 避免重复处理同一个根文件
    if root_name in processed_roots:
        continue
        
    processed_roots.add(root_name)
    
    # 检查该类别是否属于Tabletop Object
    is_tabletop = False
    if category in category_type:
        # 处理列表和字符串两种情况
        if isinstance(category_type[category], list):
            is_tabletop = "Tabletop Object" in category_type[category]
        else:
            is_tabletop = category_type[category] == "Tabletop Object"
    
    # 如果是Tabletop Object，复制所有相关文件
    if is_tabletop:
        tabletop_categories.add(category)
        # 查找并复制所有相关文件
        for related_file in os.listdir(src_dir):
            if related_file.startswith(root_name):
                src_path = os.path.join(src_dir, related_file)
                dst_path = os.path.join(dest_dir, related_file)
                shutil.copy2(src_path, dst_path)
                kept_files += 1
                print(f"复制: {related_file}")
    else:
        other_categories.add(category)

# 输出统计信息
print("\n筛选结果统计:")
print(f"总共处理了 {len(root_to_category)} 个不同对象")
print(f"保留了 {kept_files} 个文件")
print(f"保留的类别数量: {len(tabletop_categories)}")
print(f"保留的类别: {sorted(list(tabletop_categories))}")
print(f"被过滤的类别: {sorted(list(other_categories))}")

# 写入类别列表到文件
with open(os.path.join(dest_dir, 'kept_categories.txt'), 'w') as f:
    for category in sorted(list(tabletop_categories)):
        f.write(f"{category}\n")