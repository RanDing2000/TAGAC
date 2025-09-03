import os

def create_urdf_files(directory):
    # 获取目录中的所有文件
    files = os.listdir(directory)
    
    # 过滤出碰撞和纹理文件
    collision_files = [f for f in files if '_collision.obj' in f]
    textured_files = [f for f in files if '_textured.obj' in f]
    
    for collision_file in collision_files:
        # 获取对象名称
        object_name = collision_file.replace('_collision.obj', '')
        
        # 找到对应的纹理文件
        textured_file = f"{object_name}_textured.obj"
        if textured_file not in textured_files:
            print(f"Warning: No textured file found for {object_name}")
            continue
        
        # 创建URDF内容
        urdf_content = f"""<?xml version="1.0"?>
<robot name="{object_name}">
  <link name="base_link">
    <contact>
      <lateral_friction value="0.5"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="0.2"/>
    </contact>
    <inertial>
      <mass value="0.5"/> <!-- Adjust mass as needed -->
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/> <!-- Adjust inertia tensor -->
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{textured_file}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{collision_file}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        # 写入URDF文件
        urdf_filename = os.path.join(directory, f"{object_name}.urdf")
        with open(urdf_filename, 'w') as urdf_file:
            urdf_file.write(urdf_content)
        print(f"Created URDF file: {urdf_filename}")

# 使用函数
create_urdf_files('/home/ran.ding/projects/TARGO/data//maniskill_ycb/mani_skill2_ycb/collisions')