import pandas as pd
import numpy as np
import os
import json

def analyze_results(file_path):
    # 读取YCB类别映射
    ycb_path = "/home/ran.ding/projects/TARGO/targo_eval_results/eval_results_train_full-medium-occlusion-1000/targo/0/ycb.json"
    with open(ycb_path, 'r') as f:
        ycb_mapping = json.load(f)
    
    # 创建数据存储字典
    stats = {}
    total_count = 0
    
    # 读取文件
    with open(file_path, 'r') as f:
        # 跳过头行
        next(f, None)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 解析每行数据
            parts = line.split(', ')
            if len(parts) != 6:
                continue
                
            scene_id, target_label, occlusion_level, iou, cd, success = parts
            
            # 将ID映射到YCB类别
            if target_label in ycb_mapping:
                category = ycb_mapping[target_label]
            else:
                continue  # 跳过未知类别
            
            try:
                iou = float(iou)
                cd = float(cd)
                success = int(success)
            except ValueError:
                continue
            
            # 更新统计信息
            if category not in stats:
                stats[category] = {'iou_sum': 0.0, 'cd_sum': 0.0, 'count': 0, 'success_count': 0}
            
            stats[category]['iou_sum'] += iou
            stats[category]['cd_sum'] += cd
            stats[category]['count'] += 1
            stats[category]['success_count'] += success
            total_count += 1
    
    # 计算平均值并创建结果列表
    results = []
    for category, data in stats.items():
        count = data['count']
        if count > 0:
            avg_iou = data['iou_sum'] / count
            avg_cd = data['cd_sum'] / count
            success_rate = data['success_count'] / count
            results.append({
                'target_label': category,  # 使用YCB类别名称
                'avg_iou': avg_iou,
                'avg_cd': avg_cd,
                'success_rate': success_rate,
                'count': count
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 创建输出字符串
    output = []
    output.append(f"Total processed samples: {total_count}\n")
    output.append("Statistics by YCB category:")  # 修改标题
    output.append(f"{'Category':<25} {'Avg IoU':<12} {'Avg CD':<12} {'Success Rate':<12} {'Count':<8}")
    output.append("-" * 70)
    
    # 按不同指标排序
    output.append("\nSorted by IoU (descending):")
    sorted_df = df.sort_values('avg_iou', ascending=False)
    for _, row in sorted_df.iterrows():
        output.append(f"{row['target_label']:<25} {row['avg_iou']:.6f}    {row['avg_cd']:.6f}    {row['success_rate']:.6f}    {row['count']:<8}")
    
    output.append("\nSorted by CD (ascending):")
    sorted_df = df.sort_values('avg_cd', ascending=True)
    for _, row in sorted_df.iterrows():
        output.append(f"{row['target_label']:<25} {row['avg_iou']:.6f}    {row['avg_cd']:.6f}    {row['success_rate']:.6f}    {row['count']:<8}")
    
    output.append("\nSorted by Success Rate (descending):")
    sorted_df = df.sort_values('success_rate', ascending=False)
    for _, row in sorted_df.iterrows():
        output.append(f"{row['target_label']:<25} {row['avg_iou']:.6f}    {row['avg_cd']:.6f}    {row['success_rate']:.6f}    {row['count']:<8}")
    
    # 计算总体平均值
    output.append("\nOverall averages across all categories:")  # 修改标题
    output.append(f"Average IoU: {df['avg_iou'].mean():.6f}")
    output.append(f"Average CD: {df['avg_cd'].mean():.6f}")
    output.append(f"Average Success Rate: {df['success_rate'].mean():.6f}")
    
    # 确保输出目录存在
    os.makedirs("/home/ran.ding/projects/TARGO/targo_eval_results/eval_results_train_full-medium-occlusion-1000/targo/0", exist_ok=True)
    
    # 写入文件
    with open("/home/ran.ding/projects/TARGO/targo_eval_results/eval_results_train_full-medium-occlusion-1000/targo/0/summary_by_category.txt", 'w') as f:
        f.write('\n'.join(output))
    
    # 同时打印到控制台
    print('\n'.join(output))

# 使用方法
file_path = "/home/ran.ding/projects/TARGO/targo_eval_results/eval_results_train_full-medium-occlusion-1000/targo/2025-03-25_00-08-19/meta_evaluations.txt"
analyze_results(file_path)