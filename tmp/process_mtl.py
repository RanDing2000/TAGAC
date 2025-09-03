import os
import re
from glob import glob

# 定义目录路径
center_dir = "/home/ran.ding/projects/TARGO/data//acronym/ShapeNetSem-backup/models-OBJ/models_center"

def update_texture_paths(mtl_file):
    with open(mtl_file, 'r') as f:
        content = f.read()
    
    # 更新纹理路径，将 "../models-textures/textures/" 替换为 "../../models-textures/textures/"
    new_content = re.sub(
        r'map_[KkDd][ad]\s+\.\./models-textures/textures/',
        'map_Ka ../../models-textures/textures/',
        content
    )
    new_content = re.sub(
        r'map_[KkDd][ad]\s+\.\./models-textures/textures/',
        'map_Kd ../../models-textures/textures/',
        new_content
    )
    
    with open(mtl_file, 'w') as f:
        f.write(new_content)

# 统计变量
total_files = 0
modified_files = 0
errors = 0

print("开始更新MTL文件中的纹理路径...")

# 遍历所有MTL文件
for mtl_file in glob(os.path.join(center_dir, "*.mtl")):
    total_files += 1
    try:
        update_texture_paths(mtl_file)
        modified_files += 1
        
        if total_files % 20 == 0:
            print(f"已处理: {total_files}, 修改: {modified_files}, 错误: {errors}")
            
    except Exception as e:
        errors += 1
        print(f"处理文件 {mtl_file} 时出错: {str(e)}")

print("\n处理完成!")
print(f"总文件数: {total_files}")
print(f"修改的文件数: {modified_files}")
print(f"错误数: {errors}")