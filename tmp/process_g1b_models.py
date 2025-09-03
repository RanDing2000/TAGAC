import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
import trimesh

def create_urdf(model_name, output_path):
    """为指定模型创建URDF文件"""
    
    # 创建XML结构
    robot = ET.Element('robot', name=model_name)
    
    # 添加base_link
    link = ET.SubElement(robot, 'link', name='base_link')
    
    # 添加contact属性
    contact = ET.SubElement(link, 'contact')
    ET.SubElement(contact, 'lateral_friction', value='1.0')
    ET.SubElement(contact, 'rolling_friction', value='0.0')
    ET.SubElement(contact, 'contact_cfm', value='0.0')
    ET.SubElement(contact, 'contact_erp', value='1.0')
    
    # 添加inertial属性，质量设为0.5
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass', value='0.5')
    inertia = ET.SubElement(inertial, 'inertia', 
                           ixx='1', ixy='0', ixz='0', 
                           iyy='1', iyz='0', izz='1')
    
    # 添加visual属性
    visual = ET.SubElement(link, 'visual')
    visual_geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(visual_geometry, 'mesh', 
                 filename=f"{model_name}_visual.obj")
    
    # 添加collision属性
    collision = ET.SubElement(link, 'collision')
    collision_geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(collision_geometry, 'mesh', 
                 filename=f"{model_name}_collision.obj")
    
    # 输出为格式化的XML
    rough_string = ET.tostring(robot, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write(pretty_xml)

def convert_ply_to_obj(input_file, output_file):
    """将PLY文件转换为OBJ文件"""
    try:
        mesh = trimesh.load(input_file)
        mesh.export(output_file)
        return True
    except Exception as e:
        print(f"转换错误: {e}")
        return False

def main():
    # 设置路径
    base_path = "/home/ran.ding/projects/TARGO/data//g1b"
    models_dir = os.path.join(base_path, "models")
    collisions_dir = os.path.join(base_path, "collisions")
    
    # 确保collisions目录存在
    os.makedirs(collisions_dir, exist_ok=True)
    
    # 遍历models目录下的所有文件夹
    for model_folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_folder)
        
        # 跳过非目录
        if not os.path.isdir(model_path):
            continue
        
        print(f"处理模型: {model_folder}")
        
        # 创建输出目录
        output_dir = os.path.join(collisions_dir, model_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # 输入PLY文件路径
        ply_file = os.path.join(model_path, "nontextured.ply")
        
        if not os.path.exists(ply_file):
            print(f"警告: {ply_file} 不存在，跳过此模型")
            continue
        
        # 输出文件路径
        visual_obj = os.path.join(output_dir, f"{model_folder}_visual.obj")
        collision_obj = os.path.join(output_dir, f"{model_folder}_collision.obj")
        urdf_file = os.path.join(output_dir, f"{model_folder}.urdf")
        
        # 转换PLY到OBJ（视觉和碰撞模型使用相同的几何数据）
        if convert_ply_to_obj(ply_file, visual_obj):
            # 复制为碰撞模型（或者也可以简化网格后作为碰撞模型）
            shutil.copy(visual_obj, collision_obj)
            
            # 创建URDF文件
            create_urdf(model_folder, urdf_file)
            
            print(f"成功处理: {model_folder}")
        else:
            print(f"处理失败: {model_folder}")

if __name__ == "__main__":
    main() 