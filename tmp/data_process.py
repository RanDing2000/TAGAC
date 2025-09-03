import os
import shutil

# 定义源和目标目录
source_dir = '/home/ran.ding/projects/TARGO/data//maniskill_ycb/mani_skill2_ycb/models'
destination_dir = '/home/ran.ding/projects/TARGO/data//maniskill_ycb/mani_skill2_ycb/collisions'

# 遍历源目录中的所有子文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        # 定义目标文件夹路径
        target_file = os.path.join(destination_dir, f"{folder_name}_textured.obj")
        
        # 定义要复制的文件路径
        source_file = os.path.join(folder_path, 'textured.obj')
        
        # 检查源文件是否存在
        if os.path.exists(source_file):
            # 复制文件
            shutil.copy(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")
        else:
            print(f"{source_file} does not exist.")