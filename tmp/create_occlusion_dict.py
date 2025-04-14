import os
import json
import glob
import re

# 定义目录路径
mesh_pose_dir = "/usr/stud/dira/GraspInClutter/targo/output/ycb/maniskill-ycb-v2-no-occlusion-1000/mesh_pose_dict"

# 创建occlusion_level_dict
occlusion_level_dict = {}

# 遍历所有npz文件
for npz_file in glob.glob(os.path.join(mesh_pose_dir, "*.npz")):
    # 获取文件名（不含扩展名）
    filename = os.path.basename(npz_file)
    base_name = os.path.splitext(filename)[0]
    
    # 使用正则表达式提取ID
    match = re.match(r"(.+)_c_(\d+)", base_name)
    if match:
        # 将所有occlusion_level设置为0.0
        occlusion_level_dict[base_name] = 0.0

# 将字典保存为JSON文件
output_file = "/usr/stud/dira/GraspInClutter/targo/output/ycb/maniskill-ycb-v2-no-occlusion-1000/occlusion_level_dict.json"
with open(output_file, 'w') as f:
    json.dump(occlusion_level_dict, f, indent=2)

print(f"已处理 {len(occlusion_level_dict)} 个文件")
print(f"结果已保存到 {output_file}") 