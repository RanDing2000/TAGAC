import os
import xml.etree.ElementTree as ET
import re

# 定义目标目录
target_dir = "/home/ran.ding/projects/TARGO/data//acronym/collisions_tabletop"

# 统计变量
total_files = 0
modified_files = 0
skipped_files = 0
errors = 0

# 遍历目标目录中的所有文件
print(f"开始处理目录: {target_dir}")
for filename in os.listdir(target_dir):
    if filename.endswith('.urdf'):
        total_files += 1
        file_path = os.path.join(target_dir, filename)
        
        try:
            # 方法1：使用ElementTree解析和修改XML
            try:
                # 解析XML文件
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # 查找所有mass元素
                mass_elements = root.findall(".//mass")
                if mass_elements:
                    for mass in mass_elements:
                        # 获取并打印原始值
                        original_value = mass.get('value')
                        # 设置新值
                        mass.set('value', '0.5')
                        
                    # 保存修改后的文件
                    tree.write(file_path)
                    modified_files += 1
                    print(f"已修改 {filename}: 质量从 {original_value} 改为 0.5")
                else:
                    # 如果ElementTree解析找不到mass元素，尝试使用正则表达式
                    raise Exception("未找到mass元素，尝试正则表达式方法")
                    
            except Exception as e:
                # 方法2：使用正则表达式替换（应对非标准XML）
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 使用正则表达式匹配和替换质量值
                pattern = r'<mass\s+value="([^"]+)"/>'
                match = re.search(pattern, content)
                
                if match:
                    original_value = match.group(1)
                    new_content = re.sub(pattern, '<mass value="0.5"/>', content)
                    
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    
                    modified_files += 1
                    print(f"已修改 {filename} (使用正则表达式): 质量从 {original_value} 改为 0.5")
                else:
                    skipped_files += 1
                    print(f"警告: 无法找到 {filename} 中的质量值")
        
        except Exception as e:
            errors += 1
            print(f"错误: 处理 {filename} 时出错: {str(e)}")

# 输出统计信息
print("\n处理完成!")
print(f"总文件数: {total_files}")
print(f"已修改: {modified_files}")
print(f"已跳过: {skipped_files}")
print(f"错误: {errors}")