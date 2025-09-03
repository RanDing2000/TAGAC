import os
import json
import shutil

# 定义路径
urdf_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"
category_json = "/home/ran.ding/projects/TARGO/data//acronym/category_type.json"
backup_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym_backup"

# 确保备份目录存在
os.makedirs(backup_dir, exist_ok=True)

# 加载类别类型数据
with open(category_json, 'r') as f:
    category_types = json.load(f)

# 统计变量
total_files = 0
tabletop_files = 0
removed_files = 0
errors = 0

print(f"开始处理URDF文件...")

# 遍历所有URDF文件
for filename in os.listdir(urdf_dir):
    if not filename.endswith(".urdf"):
        continue
    
    total_files += 1
    file_path = os.path.join(urdf_dir, filename)
    
    try:
        # 从文件名提取类别名
        if "_" not in filename:
            print(f"警告: 文件名 {filename} 不符合预期格式 (无下划线分隔符)，跳过")
            continue
            
        category = filename.split("_")[0]
        
        # 检查类别是否在json中
        if category not in category_types:
            print(f"警告: 类别 '{category}' 在category_type.json中不存在，跳过文件 {filename}")
            continue
        
        # 获取类别类型
        type_info = category_types[category]
        
        # 检查是否为Tabletop Object
        is_tabletop = False
        if isinstance(type_info, str):
            is_tabletop = (type_info == "Tabletop Object")
        elif isinstance(type_info, list):
            is_tabletop = ("Tabletop Object" in type_info)
        
        if is_tabletop:
            # 是台面物体类别，保留
            tabletop_files += 1
            if total_files % 100 == 0:
                print(f"已处理 {total_files} 个文件，保留 {tabletop_files} 个台面物体文件")
        else:
            # 不是台面物体类别，备份并删除
            backup_path = os.path.join(backup_dir, filename)
            shutil.copy2(file_path, backup_path)  # 先备份
            os.remove(file_path)  # 再删除
            removed_files += 1
            print(f"删除非台面物体文件: {filename} (类别: {category})")
    
    except Exception as e:
        errors += 1
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总URDF文件数: {total_files}")
print(f"保留的台面物体文件数: {tabletop_files}")
print(f"删除的非台面物体文件数: {removed_files}")
print(f"处理出错文件数: {errors}")