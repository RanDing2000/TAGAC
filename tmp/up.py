import os
import re

# 定义要处理的目录
urdf_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"

# 定义要替换的路径前缀
old_prefix = "../ShapeNetSem-backup/models-OBJ/models/"
new_prefix = "ShapeNetSem-backup/models-OBJ/models/"

# 统计变量
total_files = 0
modified_files = 0
modified_paths = 0
errors = 0

print(f"开始处理目录: {urdf_dir}")

# 遍历目录中的所有文件
for filename in os.listdir(urdf_dir):
    if not filename.endswith(".urdf"):
        continue
    
    total_files += 1
    file_path = os.path.join(urdf_dir, filename)
    
    try:
        # 读取URDF文件内容
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 计算原始中包含的旧路径数量
        old_path_count = content.count(old_prefix)
        
        if old_path_count == 0:
            print(f"文件 {filename} 中没有找到需要替换的路径，跳过")
            continue
        
        # 替换mesh文件路径
        new_content = content.replace(old_prefix, new_prefix)
        
        # 计算实际替换的数量
        actual_replaced = content.count(old_prefix) - new_content.count(old_prefix)
        modified_paths += actual_replaced
        
        # 保存修改后的内容
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        modified_files += 1
        
        # 每处理100个文件输出一次进度
        if total_files % 100 == 0:
            print(f"已处理 {total_files} 个文件，修改了 {modified_files} 个文件")
            
    except Exception as e:
        errors += 1
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总文件数: {total_files}")
print(f"修改的文件数: {modified_files}")
print(f"修改的路径数: {modified_paths}")
print(f"处理出错文件数: {errors}")