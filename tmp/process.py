import os
import trimesh
import numpy as np
from glob import glob
from pathlib import Path

# 定义源目录和目标目录
source_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models_watertight"
target_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models_watertight_center"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 统计变量
total_files = 0
processed_files = 0
errors = 0

print(f"开始处理OBJ文件...")

# 获取所有OBJ文件
obj_files = glob(os.path.join(source_dir, "*.obj"))

# 处理每个OBJ文件
for obj_file in obj_files:
    total_files += 1
    file_name = os.path.basename(obj_file)
    target_file = os.path.join(target_dir, file_name)
    
    try:
        # 如果目标文件已存在，跳过处理
        if os.path.exists(target_file):
            print(f"文件已存在，跳过: {file_name}")
            processed_files += 1
            continue
        
        # 加载模型
        mesh = trimesh.load(obj_file)
        
        # 创建模型副本并居中到质心
        centered_mesh = mesh.copy()
        centered_mesh.vertices -= centered_mesh.center_mass
        
        # 保存居中后的模型
        centered_mesh.export(target_file)
        
        processed_files += 1
        
        # 输出进度
        if total_files % 20 == 0:
            print(f"已处理: {processed_files}/{total_files} 文件")
            
    except Exception as e:
        errors += 1
        print(f"处理文件 {file_name} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总文件数: {total_files}")
print(f"成功处理的文件数: {processed_files}")
print(f"处理出错的文件数: {errors}")