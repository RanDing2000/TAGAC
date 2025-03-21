import os
import re

# 定义MTL文件所在目录
mtl_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models"

# 定义要替换的路径前缀
old_prefix = "ShapeNetSem-backup/models-textures/textures/"
new_prefix = "acronym/ShapeNetSem-backup/models-textures/textures/"

# 统计变量
total_files = 0
modified_files = 0
modified_paths = 0
errors = 0

print(f"开始处理目录: {mtl_dir}")

# 遍历目录中的所有MTL文件
for filename in os.listdir(mtl_dir):
    if not filename.endswith(".mtl"):
        continue
    
    total_files += 1
    file_path = os.path.join(mtl_dir, filename)
    file_modified = False
    
    try:
        # 读取MTL文件内容
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 处理每一行，查找并替换纹理路径
        new_lines = []
        file_paths_modified = 0
        
        for line in lines:
            # 检查行是否包含纹理路径
            if line.strip().startswith("map_") and old_prefix in line:
                # 使用新前缀替换旧前缀
                new_line = line.replace(old_prefix, new_prefix)
                new_lines.append(new_line)
                file_paths_modified += 1
            else:
                new_lines.append(line)
        
        # 如果文件被修改，则保存
        if file_paths_modified > 0:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            modified_files += 1
            modified_paths += file_paths_modified
            file_modified = True
        
        # 打印处理进度
        if total_files % 100 == 0 or file_modified:
            print(f"处理文件: {filename} - {'已修改' if file_modified else '无需修改'} ({file_paths_modified} 个路径)")
        
    except Exception as e:
        errors += 1
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总MTL文件数: {total_files}")
print(f"修改的文件数: {modified_files}")
print(f"修改的路径数: {modified_paths}")
print(f"处理出错文件数: {errors}")