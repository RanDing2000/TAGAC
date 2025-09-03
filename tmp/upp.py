import os
import re
from glob import glob

# 定义目录路径
urdf_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"
backup_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym_backup_mesh"

# 确保备份目录存在
os.makedirs(backup_dir, exist_ok=True)

# 统计变量
total_files = 0
modified_files = 0
modified_paths = 0
errors = 0

print(f"开始处理URDF文件...")

# 遍历目录中的所有URDF文件
for urdf_file in glob(os.path.join(urdf_dir, "*.urdf")):
    filename = os.path.basename(urdf_file)
    total_files += 1
    
    try:
        # 读取URDF文件内容
        with open(urdf_file, 'r') as f:
            content = f.read()
        
        # 备份原始文件
        backup_path = os.path.join(backup_dir, filename)
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(content)
        
        # 查找并替换模型路径
        # 原路径: ShapeNetSem-backup/models-OBJ/models/文件ID.obj
        # 新路径: ShapeNetSem-backup/models-OBJ/models_watertight/文件ID.obj
        old_pattern = r'(ShapeNetSem-backup/models-OBJ/)models/([a-zA-Z0-9]+\.obj)'
        new_pattern = r'\1models_watertight/\2'
        
        # 计算替换前的出现次数
        old_count = len(re.findall(old_pattern, content))
        
        if old_count == 0:
            # 如果没有找到匹配项，尝试另一种可能的格式
            # 有时路径可能是相对路径的变形
            alt_pattern = r'(filename=")([^"]*models/)([a-zA-Z0-9]+\.obj)'
            alt_replacement = r'\1\2../models_watertight/\3'
            
            if re.search(alt_pattern, content):
                new_content = re.sub(alt_pattern, alt_replacement, content)
                modified = True
            else:
                print(f"警告: 在文件 {filename} 中未找到任何模型路径")
                continue
        else:
            # 执行替换
            new_content = re.sub(old_pattern, new_pattern, content)
            modified = True
        
        # 计算实际替换的次数
        new_count = len(re.findall(r'(ShapeNetSem-backup/models-OBJ/)models_watertight/([a-zA-Z0-9]+\.obj)', new_content))
        paths_modified = new_count if new_count > 0 else 0
        
        # 保存修改后的文件
        if modified:
            with open(urdf_file, 'w') as f:
                f.write(new_content)
            
            modified_files += 1
            modified_paths += paths_modified
            
            # 打印处理信息
            if total_files % 10 == 0 or paths_modified > 0:
                print(f"处理文件: {filename} - 修改了 {paths_modified} 个路径")
    
    except Exception as e:
        errors += 1
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总URDF文件数: {total_files}")
print(f"修改的文件数: {modified_files}")
print(f"修改的路径数: {modified_paths}")
print(f"处理出错文件数: {errors}")