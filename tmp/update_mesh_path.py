import os
import re
from glob import glob

# 定义目录路径
urdf_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"

# 统计变量
total_files = 0
modified_files = 0
errors = 0

print(f"开始处理URDF文件...")

# 遍历所有URDF文件
for urdf_file in glob(os.path.join(urdf_dir, "*.urdf")):
    total_files += 1
    filename = os.path.basename(urdf_file)
    
    try:
        # 读取URDF文件内容
        with open(urdf_file, 'r') as f:
            content = f.read()
        
        # 提取mesh ID (a924eb3037129eaff8095890d92b7d6c)
        mesh_id = re.search(r'([a-f0-9]{32})\.urdf$', filename)
        if mesh_id:
            mesh_id = mesh_id.group(1)
            
            # 替换mesh路径
            pattern = r'filename="[^"]*models_watertight/[^"]*"'
            new_path = f'filename="ShapeNetSem-backup/models-OBJ/models/{mesh_id}.obj"'
            
            new_content = re.sub(pattern, new_path, content)
            
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