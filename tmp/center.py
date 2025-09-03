import os
import re
from glob import glob

# Define directory path
urdf_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"

# Statistics variables
total_files = 0
modified_files = 0
errors = 0

print(f"开始处理URDF文件...")

# Traverse all URDF files
for urdf_file in glob(os.path.join(urdf_dir, "*.urdf")):
    total_files += 1
    filename = os.path.basename(urdf_file)
    
    try:
        # Read URDF file content
        with open(urdf_file, 'r') as f:
            content = f.read()
        
        # Extract mesh ID (a924eb3037129eaff8095890d92b7d6c)
        mesh_id_match = re.search(r'([a-f0-9]{32})\.urdf$', filename)
        # Also handle shorter IDs like e3694ae657167c79bec436d295978d
        if not mesh_id_match:
            mesh_id_match = re.search(r'([a-f0-9]{30,31})\.urdf$', filename)
        
        if mesh_id_match:
            mesh_id = mesh_id_match.group(1)
            
            # Replace visual mesh path - handle models path
            visual_pattern = r'(<visual>.*?<mesh filename=")ShapeNetSem-backup/models-OBJ/models/[^"]*(".*?/>)'
            visual_replacement = rf'\1ShapeNetSem-backup/models-OBJ/models_center/{mesh_id}.obj\2'
            
            # Replace collision mesh path - handle models path
            collision_pattern = r'(<collision>.*?<mesh filename=")ShapeNetSem-backup/models-OBJ/models/[^"]*(".*?/>)'
            collision_replacement = rf'\1ShapeNetSem-backup/models-OBJ/models_watertight_center/{mesh_id}.obj\2'
            
            # Replace visual mesh path - handle models_watertight path
            visual_watertight_pattern = r'(<visual>.*?<mesh filename=")ShapeNetSem-backup/models-OBJ/models_watertight/[^"]*(".*?/>)'
            visual_watertight_replacement = rf'\1ShapeNetSem-backup/models-OBJ/models_center/{mesh_id}.obj\2'
            
            # Replace collision mesh path - handle models_watertight path
            collision_watertight_pattern = r'(<collision>.*?<mesh filename=")ShapeNetSem-backup/models-OBJ/models_watertight/[^"]*(".*?/>)'
            collision_watertight_replacement = rf'\1ShapeNetSem-backup/models-OBJ/models_watertight_center/{mesh_id}.obj\2'
            
            # Apply replacements
            new_content = re.sub(visual_pattern, visual_replacement, content, flags=re.DOTALL)
            new_content = re.sub(collision_pattern, collision_replacement, new_content, flags=re.DOTALL)
            new_content = re.sub(visual_watertight_pattern, visual_watertight_replacement, new_content, flags=re.DOTALL)
            new_content = re.sub(collision_watertight_pattern, collision_watertight_replacement, new_content, flags=re.DOTALL)
            
            # If content has changed, save the file
            if new_content != content:
                with open(urdf_file, 'w') as f:
                    f.write(new_content)
                modified_files += 1
                
                # Output progress every 10 files
                if total_files % 10 == 0:
                    print(f"已处理 {total_files} 个文件，修改了 {modified_files} 个文件")
        else:
            print(f"无法从文件名 {filename} 提取mesh ID")
        
    except Exception as e:
        errors += 1
        print(f"处理文件 {filename} 时出错: {str(e)}")

# Output statistics
print("\n处理完成!")
print(f"总URDF文件数: {total_files}")
print(f"修改的文件数: {modified_files}")
print(f"处理出错文件数: {errors}")