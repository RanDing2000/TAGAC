import os
import numpy as np
from pathlib import Path

def centralize_obj_vertices(mtl_file):
    # 获取mesh ID
    mesh_id = os.path.splitext(os.path.basename(mtl_file))[0]
    
    # 构建源obj和目标obj的路径
    source_obj = f"/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models/{mesh_id}.obj"
    target_obj = f"/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models_center/{mesh_id}.obj"
    
    if not os.path.exists(source_obj):
        print(f"找不到源文件: {source_obj}")
        return False
    
    # 读取OBJ文件
    with open(source_obj, 'r') as f:
        lines = f.readlines()
    
    # 收集所有顶点
    vertices = []
    for line in lines:
        if line.startswith('v '):
            coords = [float(x) for x in line.strip().split()[1:4]]
            vertices.append(coords)
    
    if not vertices:
        print(f"没有找到顶点数据: {source_obj}")
        return False
    
    # 计算中心点
    vertices = np.array(vertices)
    center = np.mean(vertices, axis=0)
    
    # 写入新的OBJ文件
    with open(target_obj, 'w') as f:
        vertex_index = 0
        for line in lines:
            if line.startswith('v '):
                # 处理顶点行
                v = vertices[vertex_index] - center
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                vertex_index += 1
            else:
                # 其他行保持不变
                f.write(line)
    
    return True

# 主处理流程
center_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/ShapeNetSem-backup/models-OBJ/models_center"

# 统计变量
total_files = 0
processed_files = 0
errors = 0

print("开始处理OBJ文件...")

# 遍历所有MTL文件
for mtl_file in Path(center_dir).glob("*.mtl"):
    total_files += 1
    try:
        if centralize_obj_vertices(mtl_file):
            processed_files += 1
            if processed_files % 20 == 0:
                print(f"已处理: {processed_files}/{total_files}")
    except Exception as e:
        errors += 1
        print(f"处理文件 {mtl_file.name} 时出错: {str(e)}")

print("\n处理完成!")
print(f"总文件数: {total_files}")
print(f"成功处理: {processed_files}")
print(f"错误数: {errors}")