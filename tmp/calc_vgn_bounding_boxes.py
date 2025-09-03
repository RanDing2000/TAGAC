#!/usr/bin/env python3
import os
import json
import numpy as np
import trimesh
from pathlib import Path

# 输入和输出路径
base_path = "/home/ran.ding/projects/TARGO/data//urdfs/packed/train"
output_dir = "/home/ran.ding/projects/TARGO/data//stastics/vgn"
output_file = os.path.join(output_dir, "vgn_bounding_boxes.json")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 结果字典
results = {}

# 处理所有碰撞模型文件
for obj_file in Path(base_path).glob("*_collision.obj"):
    try:
        # 从文件名中提取模型名称
        model_name = obj_file.stem.replace("_collision", "")
        
        # 加载碰撞网格
        mesh = trimesh.load_mesh(str(obj_file))
        
        # 计算边界框大小
        bounds = mesh.bounds
        bbox_size = (bounds[1] - bounds[0]).tolist()
        
        # 存储结果
        results[model_name] = bbox_size
        
        print(f"处理 {model_name}: 边界框大小 = {bbox_size}")
    except Exception as e:
        print(f"处理 {obj_file.name} 时出错: {e}")

# 写入JSON文件
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"结果已保存到 {output_file}")