#!/usr/bin/env python3
"""
计算YCB对象的边界框大小并保存到JSON文件中
"""

import os
import json
import numpy as np
import trimesh
from pathlib import Path
import argparse

def calculate_bounding_box_size(mesh):
    """计算模型的边界框大小(长宽高)"""
    bounds = mesh.bounds  # 包含min_bound和max_bound，形状为(2, 3)
    bounding_box_size = bounds[1] - bounds[0]  # 计算差值得到边界框大小
    return bounding_box_size.tolist()  # 转换为Python列表以便JSON序列化

def process_ycb_objects(base_path):
    """处理所有YCB对象模型，计算它们的边界框大小"""
    # 基本路径
    models_path = Path(base_path)
    
    # 结果字典
    results = {}
    
    # 遍历所有模型文件夹
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            collision_obj_path = model_dir / "collision.obj"
            
            if collision_obj_path.exists():
                try:
                    # 加载碰撞网格
                    mesh = trimesh.load_mesh(str(collision_obj_path))
                    
                    # 计算边界框大小
                    bbox_size = calculate_bounding_box_size(mesh)
                    
                    # 存储结果
                    results[model_name] = bbox_size
                    
                    print(f"处理 {model_name}: 边界框大小 = {bbox_size}")
                except Exception as e:
                    print(f"处理 {model_name} 时出错: {e}")
            else:
                print(f"找不到碰撞模型文件: {collision_obj_path}")
    
    return results

def main():
    # 输出路径
    output_dir = "/usr/stud/dira/GraspInClutter/targo/data/stastics/maniskill_ycb"
    output_file = os.path.join(output_dir, "ycb_bounding_boxes.json")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # YCB模型的基本路径
    base_path = "/usr/stud/dira/GraspInClutter/targo/data/maniskill_ycb/mani_skill2_ycb/models"
    
    # 处理所有模型
    results = process_ycb_objects(base_path)
    
    # 写入JSON文件
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    main()