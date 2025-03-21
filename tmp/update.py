import os
import re

# 定义目标目录
target_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/collisions_tabletop"

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
            # 匹配形如AAABattery_a924eb3037129eaff8095890d92b7d6c_visual.obj或AAABattery_a924eb3037129eaff8095890d92b7d6c_collision.obj的模式
            path_pattern = r'filename="([^"]+/)?([^_]+)_([a-f0-9]+)_(?:visual|collision)\.obj"'
            
            # 记录原始内容以检查是否有变化
            original_content = content
            
            # 执行替换，将文件名部分改为ID.obj
            modified_content = re.sub(path_pattern, r'filename="\1\3.obj"', content)
            
            # 检查是否有变化
            if modified_content != original_content:
                # 计算替换次数
                path_count = len(re.findall(path_pattern, content))
                
                # 保存修改后的文件
                with open(file_path, 'w') as f:
                    f.write(modified_content)
                
                modified_files += 1
                modified_paths += path_count
                print(f"已修改 {filename}: {path_count} 个路径")
            else:
                print(f"跳过 {filename}: 没有找到匹配的路径")
            
        except Exception as e:
            errors += 1
            print(f"错误: 处理 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总URDF文件数: {total_files}")
print(f"已修改文件数: {modified_files}")
print(f"修改的路径总数: {modified_paths}")
print(f"处理出错文件数: {errors}")