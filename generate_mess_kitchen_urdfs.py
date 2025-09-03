#!/usr/bin/env python3
import os
from pathlib import Path

def generate_urdf_for_object(object_name, mesh_path):
    """为单个对象生成URDF文件"""
    urdf_content = f'''<?xml version="1.0"?>
<robot name="{object_name}">
  <link name="base_link">
    <contact>
      <lateral_friction value="1.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </collision>
  </link>
</robot>'''
    return urdf_content

def main():
    # 获取mess_kitchen文件夹路径
    mess_kitchen_path = Path("data/gso/google_scanned_objects/mess_kitchen")
    
    if not mess_kitchen_path.exists():
        print(f"错误：找不到目录 {mess_kitchen_path}")
        return
    
    # 创建输出目录
    output_dir = Path("data/urdfs/mess_kitchen")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有子文件夹
    object_dirs = [d for d in mess_kitchen_path.iterdir() if d.is_dir()]
    
    print(f"找到 {len(object_dirs)} 个对象目录")
    
    generated_count = 0
    for obj_dir in object_dirs:
        object_name = obj_dir.name
        
        # 检查是否存在model.obj文件
        model_obj_path = obj_dir / "meshes" / "model.obj"
        
        if not model_obj_path.exists():
            print(f"警告：{object_name} 没有找到 model.obj 文件")
            continue
        
        # 生成URDF内容
        mesh_path = f"data/gso/google_scanned_objects/mess_kitchen/{object_name}/meshes/model.obj"
        urdf_content = generate_urdf_for_object(object_name, mesh_path)
        
        # 保存URDF文件
        urdf_file_path = output_dir / f"{object_name}.urdf"
        with open(urdf_file_path, 'w') as f:
            f.write(urdf_content)
        
        print(f"✓ 生成 {object_name}.urdf")
        generated_count += 1
    
    print(f"\n完成！生成了 {generated_count} 个URDF文件")
    print(f"输出目录：{output_dir}")

if __name__ == "__main__":
    main()
