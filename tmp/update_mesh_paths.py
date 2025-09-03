import os
import xml.etree.ElementTree as ET
import re

# 定义目标目录和新路径前缀
target_dir = "/home/ran.ding/projects/TARGO/data//acronym/collisions_tabletop"
new_path_prefix = "/home/ran.ding/projects/TARGO/data/acronym/ShapeNetSem-backup/models-OBJ/models/"

# 统计变量
total_files = 0
modified_files = 0
modified_paths = 0
errors = 0

# 遍历目标目录中的所有文件
print(f"开始处理目录: {target_dir}")
for filename in os.listdir(target_dir):
    if filename.endswith('.urdf'):
        total_files += 1
        file_path = os.path.join(target_dir, filename)
        
        try:
            # 读取文件内容
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 使用正则表达式查找并替换所有mesh文件路径
            path_pattern = r'filename="([^"]+\.obj)"'
            
            # 查找所有匹配项
            matches = re.findall(path_pattern, content)
            path_count = len(matches)
            
            # 对每个匹配项进行替换
            for old_path in matches:
                # 跳过已经是绝对路径的情况
                if old_path.startswith('/'):
                    continue
                
                # 构建新路径
                filename_only = os.path.basename(old_path)
                new_path = new_path_prefix + filename_only
                
                # 执行替换
                content = content.replace(f'filename="{old_path}"', f'filename="{new_path}"')
            
            # 保存修改后的文件
            with open(file_path, 'w') as f:
                f.write(content)
            
            modified_files += 1
            modified_paths += path_count
            print(f"已修改 {filename}: {path_count} 个路径")
            
        except Exception as e:
            errors += 1
            print(f"错误: 处理 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总URDF文件数: {total_files}")
print(f"已修改文件数: {modified_files}")
print(f"修改的路径总数: {modified_paths}")
print(f"处理出错文件数: {errors}")