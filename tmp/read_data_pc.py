import numpy as np
from pathlib import Path

def read_test_set_point_cloud(root, scene_id, name="mesh_pose_dict"):
    # 构建文件路径
    path = root / name / (scene_id + ".npz")
    
    # 加载 .npz 文件
    with np.load(path, allow_pickle=True) as data:
        # 读取点云数据和目标名称
        point_cloud = data['pc']
        target_name = data['target_name']
    
    return point_cloud, target_name

# 示例用法
root = Path("/usr/stud/dira/GraspInClutter/targo/output/maniskill-ycb-v2")
scene_id = "1af05ea5433c400898f18d42f19b97f0_c_1"
point_cloud, target_name = read_test_set_point_cloud(root, scene_id)

print("Point Cloud:", point_cloud)
print("Target Name:", target_name)