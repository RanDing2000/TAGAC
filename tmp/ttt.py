import os
import shutil
from glob import glob
from pathlib import Path

# 定义目录路径
center_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models_center"
source_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models"

# 统计变量
total_files = 0
copied_files = 0
errors = 0

print("开始处理MTL文件...")

# 遍历models_center中的所有obj文件
for obj_file in glob(os.path.join(center_dir, "*.obj")):
    total_files += 1
    
    try:
        # 获取mesh ID
        mesh_id = Path(obj_file).stem
        
        # 构建源mtl文件路径和目标mtl文件路径
        source_mtl = os.path.join(source_dir, f"{mesh_id}.mtl")
        target_mtl = os.path.join(center_dir, f"{mesh_id}.mtl")
        
        # 检查源mtl文件是否存在
        if os.path.exists(source_mtl):
            # 复制mtl文件
            shutil.copy2(source_mtl, target_mtl)
            copied_files += 1
            
            # 每处理20个文件输出一次进度
            if total_files % 20 == 0:
                print(f"已处理: {total_files}, 成功复制: {copied_files}, 错误: {errors}")
        else:
            print(f"找不到MTL文件: {source_mtl}")
            
    except Exception as e:
        errors += 1
        print(f"处理文件 {obj_file} 时出错: {str(e)}")

# 输出最终统计信息
print("\n处理完成!")
print(f"总文件数: {total_files}")
print(f"成功复制文件数: {copied_files}")
print(f"处理出错文件数: {errors}")