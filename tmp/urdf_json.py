import csv
import json
import os

# 输入和输出路径
csv_file = "/usr/stud/dira/GraspInClutter/targo/data/acronym/grasps_acronym_metadata.csv"
json_output = "/usr/stud/dira/GraspInClutter/targo/data/acronym/urdf_scales.json"

# 创建用于存储结果的字典
urdf_scales = {}

# 读取CSV文件
print(f"读取CSV文件: {csv_file}")
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # 跳过表头
    
    # 检查CSV格式是否符合预期
    if len(header) < 4 or header[0] != "category" or header[3] != "scale":
        print(f"警告: CSV格式可能不正确，请检查。当前表头: {header}")
    
    # 处理每一行
    row_count = 0
    for row in reader:
        try:
            if len(row) < 4:
                print(f"警告: 行 {row_count+1} 格式不正确，跳过: {row}")
                continue
                
            category = row[0]
            object_id = row[1]
            scale = float(row[3])  # 转换为浮点数
            
            # 创建键名
            urdf_key = f"{category}_{object_id}.urdf"
            
            # 添加到字典
            urdf_scales[urdf_key] = scale
            
            row_count += 1
            if row_count % 1000 == 0:
                print(f"已处理 {row_count} 行...")
                
        except Exception as e:
            print(f"处理行时出错: {row}, 错误: {str(e)}")

# 将结果写入JSON文件
print(f"将数据写入JSON文件: {json_output}")
with open(json_output, 'w') as f:
    json.dump(urdf_scales, f, indent=2)

print(f"处理完成! 共处理 {row_count} 行，创建了 {len(urdf_scales)} 个键值对。")

# 打印前5个键值对作为示例
print("\n前5个键值对示例:")
for i, (key, value) in enumerate(list(urdf_scales.items())[:5]):
    print(f"  \"{key}\": {value}")