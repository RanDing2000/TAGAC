import os
import h5py
import pandas as pd
import re
from tqdm import tqdm

def extract_info_from_h5_files(directory_path, output_csv):
    """
    遍历目录中的所有h5文件，提取category、id、mass和scale信息，并写入CSV文件
    """
    # 存储所有提取的数据
    data = []
    
    # 获取目录中的所有h5文件
    h5_files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]
    print(f"找到 {len(h5_files)} 个H5文件")
    
    # 使用tqdm显示进度条
    for filename in tqdm(h5_files, desc="处理H5文件"):
        try:
            # 从文件名中提取category和id
            # 假设格式为：category_id_number.h5
            parts = filename.split('_')
            if len(parts) >= 3:
                category = parts[0]
                obj_id = parts[1]
                
                # 完整文件路径
                file_path = os.path.join(directory_path, filename)
                
                # 打开H5文件并读取mass和scale
                with h5py.File(file_path, 'r') as f:
                    # 检查所需的数据集是否存在
                    if '/object/mass' in f and '/object/scale' in f:
                        mass = f['/object/mass'][()]
                        scale = f['/object/scale'][()]
                        
                        # 添加到数据列表
                        data.append({
                            'category': category,
                            'id': obj_id,
                            'mass': float(mass),
                            'scale': float(scale)
                        })
                    else:
                        print(f"警告: 文件 {filename} 中缺少必要的数据集")
            else:
                print(f"警告: 文件名 {filename} 格式不正确，无法提取category和id")
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV
    df.to_csv(output_csv, index=False)
    print(f"已成功将 {len(data)} 条记录写入 {output_csv}")
    
    # 显示数据样本
    if not df.empty:
        print("\n数据样本 (前5行):")
        print(df.head())
    
    return df

if __name__ == "__main__":
    # 指定目录和输出文件
    directory_path = "/usr/stud/dira/GraspInClutter/acronym/data/grasps"
    output_csv = "grasps_metadata.csv"
    
    # 执行提取
    df = extract_info_from_h5_files(directory_path, output_csv)