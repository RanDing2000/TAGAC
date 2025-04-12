import os
import re
from glob import glob

# 定义目录路径
urdf_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/urdfs_acronym"

# 统计变量
total_files = 0
modified_files = 0
errors = 0

print("开始处理URDF文件...")

# 遍历所有URDF文件
for urdf_file in glob(os.path.join(urdf_dir, "*.urdf")):
    total_files += 1
    filename = os.path.basename(urdf_file)
    
    try:
        # 读取URDF文件内容
        with open(urdf_file, 'r') as f:
            content = f.read()
        
        # 替换mesh路径
        new_content = content.replace("models_watertight", "models_center")
        
        # 如果内容有变化，保存文件
        if new_content != content:
            with open(urdf_file, 'w') as f:
                f.write(new_content)
            modified_files += 1
            
            # 每处理10个文件输出一次进度
            if total_files % 10 == 0:
                print(f"已处理 {total_files} 个文件，修改了 {modified_files} 个文件")
    
    except Exception as e:
        errors += 1
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总URDF文件数: {total_files}")
print(f"修改的文件数: {modified_files}")
print(f"处理出错文件数: {errors}")