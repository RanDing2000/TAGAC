import os
import csv
import re

# 定义输入和输出路径
csv_file = "/home/ran.ding/projects/TARGO/data//acronym/grasps_acronym_metadata.csv"
output_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"
template_file = "/home/ran.ding/projects/TARGO/data//acronym/collisions_tabletop/AAABattery_a924eb3037129eaff8095890d92b7d6c.urdf"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取模板文件
with open(template_file, 'r') as f:
    template_content = f.read()

# 统计变量
total_rows = 0
created_files = 0
errors = 0

# 读取CSV文件并创建URDF文件
print(f"开始处理CSV文件: {csv_file}")
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    # 跳过可能的表头
    try:
        first_row = next(reader)
        # 检查第一行是否为表头（如果第三列不是浮点数，则认为是表头）
        try:
            float(first_row[2])
            # 如果可以转换为浮点数，则是数据行，需要重置文件指针
            csvfile.seek(0)
            reader = csv.reader(csvfile)
        except (ValueError, IndexError):
            # 如果不能转换为浮点数，则确实是表头，继续处理
            pass
    except StopIteration:
        print("CSV文件为空")
        exit(1)
    
    # 处理每一行数据
    for row in reader:
        total_rows += 1
        
        try:
            # 解析CSV行数据
            if len(row) < 3:
                print(f"警告: 行 {total_rows} 格式不正确，跳过")
                continue
                
            category = row[0]
            object_id = row[1]
            mass = float(row[2])  # 转换为浮点数
            
            # 创建URDF文件名
            urdf_filename = f"{category}_{object_id}.urdf"
            urdf_path = os.path.join(output_dir, urdf_filename)
            
            # 复制模板并修改内容
            urdf_content = template_content
            
            # 1. 修改robot名称
            urdf_content = re.sub(r'name="[^"]+"', f'name="{category}_{object_id}"', urdf_content)
            
            # 2. 修改质量值
            urdf_content = re.sub(r'<mass value="[^"]+" />', f'<mass value="{mass}" />', urdf_content)
            
            # 3. 修改mesh路径
            mesh_path = f'/home/ran.ding/projects/TARGO/data/acronym/ShapeNetSem-backup/models-OBJ/models/{object_id}.obj'
            urdf_content = re.sub(r'filename="[^"]+"', f'filename="{mesh_path}"', urdf_content)
            
            # 保存URDF文件
            with open(urdf_path, 'w') as f:
                f.write(urdf_content)
            
            created_files += 1
            if total_rows % 100 == 0:
                print(f"已处理 {total_rows} 行，创建了 {created_files} 个URDF文件")
            
        except Exception as e:
            errors += 1
            print(f"错误: 处理第 {total_rows} 行时出错: {str(e)}")
            print(f"  行内容: {row}")

# 输出统计信息
print("\n处理完成!")
print(f"总行数: {total_rows}")
print(f"创建的URDF文件数: {created_files}")
print(f"处理出错行数: {errors}")