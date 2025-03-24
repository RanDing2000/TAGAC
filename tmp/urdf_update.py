import os
import re
from glob import glob

# 定义目录路径
urdf_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/urdfs_acronym"
backup_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/urdfs_acronym"

# 确保备份目录存在
os.makedirs(backup_dir, exist_ok=True)

# 统计变量
total_files = 0
modified_files = 0
errors = 0

print(f"开始处理URDF文件...")

# 遍历所有URDF文件
for urdf_file in glob(os.path.join(urdf_dir, "*.urdf")):
    filename = os.path.basename(urdf_file)
    total_files += 1
    
    try:
        # 读取URDF文件内容
        with open(urdf_file, 'r') as f:
            content = f.read()
        
        # 备份原始文件
        backup_path = os.path.join(backup_dir, filename)
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(content)
        
        # 定义正则表达式模式来匹配<inertial>部分
        # 这个模式匹配包含mass、inertia和origin的inertial标签
        pattern = r'<inertial>(.*?)<mass value="(.*?)" />(.*?)<inertia (.*?) />(.*?)<origin xyz="(.*?)" rpy="(.*?)" /></inertial>'
        
        # 检查是否匹配到上述模式
        if re.search(pattern, content, re.DOTALL):
            # 替换为新格式
            new_content = re.sub(
                pattern,
                r'<inertial>\n    <origin xyz="\6" rpy="\7" />\n    <mass value="\2" />\n    <inertia \n      \4 />\n  </inertial>',
                content,
                flags=re.DOTALL
            )
        else:
            # 尝试另一种可能的格式（没有origin的情况）
            pattern2 = r'<inertial>(.*?)<mass value="(.*?)" />(.*?)<inertia (.*?) /></inertial>'
            if re.search(pattern2, content, re.DOTALL):
                # 这种情况下只有mass和inertia，没有origin
                # 首先从内容中提取mass和inertia的值
                match = re.search(pattern2, content, re.DOTALL)
                if match:
                    mass_value = match.group(2)
                    inertia_attrs = match.group(4)
                    
                    # 提取inertia属性并格式化
                    inertia_formatted = inertia_attrs.replace(' ', '\n      ')
                    
                    # 替换为新格式，不包含origin标签
                    new_content = re.sub(
                        pattern2,
                        f'<inertial>\n    <mass value="{mass_value}" />\n    <inertia \n      {inertia_formatted} />\n  </inertial>',
                        content,
                        flags=re.DOTALL
                    )
                else:
                    print(f"警告: 无法匹配 {filename} 中的inertial部分，跳过")
                    continue
            else:
                # 尝试第三种可能的格式（顺序不同：先mass、后inertia、最后origin）
                pattern3 = r'<inertial>(.*?)<mass value="(.*?)" />(.*?)<inertia (.*?) />(.*?)<origin xyz="(.*?)" rpy="(.*?)" /></inertial>'
                
                if re.search(pattern3, content, re.DOTALL):
                    # 替换为新格式
                    new_content = re.sub(
                        pattern3,
                        r'<inertial>\n    <origin xyz="\6" rpy="\7" />\n    <mass value="\2" />\n    <inertia \n      \4 />\n  </inertial>',
                        content,
                        flags=re.DOTALL
                    )
                else:
                    # 完全自定义搜索和替换
                    # 首先提取整个inertial块
                    inertial_pattern = r'<inertial>.*?</inertial>'
                    inertial_match = re.search(inertial_pattern, content, re.DOTALL)
                    
                    if inertial_match:
                        inertial_block = inertial_match.group(0)
                        
                        # 提取各个组件
                        mass_match = re.search(r'<mass value="(.*?)" />', inertial_block)
                        inertia_match = re.search(r'<inertia (.*?) />', inertial_block)
                        origin_match = re.search(r'<origin xyz="(.*?)" rpy="(.*?)" />', inertial_block)
                        
                        if mass_match and inertia_match:
                            mass_value = mass_match.group(1)
                            inertia_attrs = inertia_match.group(1)
                            
                            # 格式化inertia属性
                            inertia_parts = []
                            for part in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                                match = re.search(f'{part}="([^"]*)"', inertia_attrs)
                                if match:
                                    inertia_parts.append(f'{part}="{match.group(1)}"')
                            
                            # 构建新的分行格式
                            ixx_ixy_ixz = " ".join(inertia_parts[:3])
                            iyy_iyz_izz = " ".join(inertia_parts[3:])
                            
                            # 构建新的inertial块
                            new_inertial = "<inertial>\n"
                            
                            # 添加origin（如果存在）
                            if origin_match:
                                xyz = origin_match.group(1)
                                rpy = origin_match.group(2)
                                new_inertial += f'    <origin xyz="{xyz}" rpy="{rpy}" />\n'
                            
                            # 添加mass和inertia
                            new_inertial += f'    <mass value="{mass_value}" />\n'
                            new_inertial += f'    <inertia \n      {ixx_ixy_ixz} \n      {iyy_iyz_izz} />\n  </inertial>'
                            
                            # 替换原来的inertial块
                            new_content = content.replace(inertial_block, new_inertial)
                        else:
                            print(f"警告: 无法在 {filename} 中提取mass或inertia值，跳过")
                            continue
                    else:
                        print(f"警告: 无法在 {filename} 中找到inertial块，跳过")
                        continue
        
        # 保存修改后的文件
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