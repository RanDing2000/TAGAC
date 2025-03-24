import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 路径设置
eval_file = "/usr/stud/dira/GraspInClutter/targo/targo_eval_results/eval_results_train_full-middle-occlusion-1000/targo_full_targ/2025-03-22_20-18-47/meta_evaluations.txt"
category_file = "/usr/stud/dira/GraspInClutter/targo/targo_eval_results/stastics_analysis/category_ycb.json"
output_dir = "/usr/stud/dira/GraspInClutter/targo/targo_eval_results/eval_results_train_full-middle-occlusion-1000/targo_full_targ/2025-03-22_20-18-47"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载类别映射
with open(category_file, 'r') as f:
    category_mapping = json.load(f)

# 初始化数据结构
target_counts = defaultdict(int)
target_success = defaultdict(int)
category_counts = defaultdict(int)
category_success = defaultdict(int)
category_targets = defaultdict(set)  # 记录每个类别包含的目标

# 新增: IoU和CD统计数据结构
target_iou_values = defaultdict(list)
target_cd_values = defaultdict(list)
category_iou_values = defaultdict(list)
category_cd_values = defaultdict(list)

# 场景ID到目标名称的映射
scene_target_map = {}

# 新增: 查找并读取包含CD和IoU的元评估文件
def find_meta_evaluation_files(base_dir):
    print(f"搜索目录: {base_dir}")
    meta_files = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in ["meta_evaluation.txt", "meta_evaluations.txt"]:
                print(f"找到文件: {os.path.join(root, file)}")
                try:
                    # 尝试从路径中提取场景ID
                    scene_id = os.path.basename(os.path.dirname(os.path.dirname(root)))
                    meta_files[scene_id] = os.path.join(root, file)
                except:
                    # 如果路径结构不符预期，使用文件路径作为键
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    meta_files[rel_path] = os.path.join(root, file)
    return meta_files

# 解析评估文件，获取场景ID和目标名称的映射并计算成功率
print("Parsing evaluation file...")
with open(eval_file, 'r') as f:
    for line in f:
        line = line.strip()
        # 跳过空行和摘要行
        if not line or 'Average' in line or 'Success' in line or 'Total' in line:
            continue
        
        # 解析行
        if '|' in line:
            # 新格式: ID| 场景ID, 物体ID, 数值1, 数值2, 数值3, 成功标志(0/1)
            parts = line.split('|')
            if len(parts) < 2:
                continue
                
            data_parts = parts[1].strip().split(', ')
            if len(data_parts) < 6:
                continue
                
            scene_id = data_parts[0].strip()
            target_name = data_parts[1].strip()
            success = int(data_parts[5].strip())
        else:
            # 旧格式: 场景ID, 物体ID, 数值1, 数值2, 数值3, 成功标志(0/1)
            parts = line.split(', ')
            if len(parts) < 6:
                print(f"Skipping line with unexpected format: {line}")
                continue
            
            scene_id = parts[0]
            target_name = parts[1]
            success = int(parts[5])
        
        try:
            # 存储场景ID到目标名称的映射
            scene_target_map[scene_id] = target_name
            
            # 更新目标计数
            target_counts[target_name] += 1
            target_success[target_name] += success
            
            # 获取类别并更新类别计数
            category = category_mapping.get(target_name, "Unknown")
            category_counts[category] += 1
            category_success[category] += success
            category_targets[category].add(target_name)
            
        except (ValueError, IndexError) as e:
            print(f"Error processing line: {line}")
            print(f"Error details: {e}")
            continue

# 新增: 尝试查找并处理包含CD和IoU的元评估文件
# meta_evaluation_files = find_meta_evaluation_files("eval_results_train_full-middle-occlusion-1000")
print("Extracting CD and IoU values...")
try:
    with open(eval_file, 'r') as f:
        content = f.read()
        
        # 遍历已处理的场景和目标
        for scene_id, target_name in scene_target_map.items():
            category = category_mapping.get(target_name, "Unknown")
            
            # 提取CD和IoU值，基于元评估文件格式
            # 参考格式: "Scene_ID, Target_Name, Occlusion_Level, IoU, CD, Success"
            pattern = rf"{scene_id},\s*{target_name},\s*[\d\.]+,\s*([\d\.]+),\s*([\d\.]+)"
            match = re.search(pattern, content)
            
            if match:
                iou = float(match.group(1))
                cd = float(match.group(2))
                
                target_cd_values[target_name].append(cd)
                target_iou_values[target_name].append(iou)
                category_cd_values[category].append(cd)
                category_iou_values[category].append(iou)
                print(f"提取到 {target_name} (类别: {category}) 的 CD={cd}, IoU={iou}")
except Exception as e:
    print(f"处理文件 {eval_file} 时出错: {e}")

# 计算每个目标的成功率
target_success_rates = {}
for target in target_counts:
    if target_counts[target] > 0:
        success_rate = target_success[target] / target_counts[target]
        target_success_rates[target] = success_rate

# 计算每个类别的成功率和平均IoU/CD
category_success_rates = {}
category_avg_iou = {}
category_avg_cd = {}

for category in category_counts:
    if category_counts[category] > 0:
        success_rate = category_success[category] / category_counts[category]
        category_success_rates[category] = success_rate
    
    # 计算平均IoU和CD
    if category_iou_values[category]:
        category_avg_iou[category] = np.mean(category_iou_values[category])
    else:
        category_avg_iou[category] = 0
        
    if category_cd_values[category]:
        category_avg_cd[category] = np.mean(category_cd_values[category])
    else:
        category_avg_cd[category] = 0

# 按成功率排序类别
sorted_categories = sorted(category_success_rates.items(), key=lambda x: x[1], reverse=True)

# 生成类别报告
with open(f"{output_dir}/category_success_rates.json", 'w') as f:
    data = {
        "success_rates": category_success_rates,
        "avg_iou": category_avg_iou,
        "avg_cd": category_avg_cd
    }
    json.dump(data, f, indent=4)

with open(f"{output_dir}/category_success_summary.txt", 'w') as f:
    f.write("类别成功率与质量统计：\n")
    f.write("="*60 + "\n\n")
    
    for i, (category, rate) in enumerate(sorted_categories, 1):
        targets_in_category = len(category_targets[category])
        iou = category_avg_iou.get(category, 0)
        cd = category_avg_cd.get(category, 0)
        
        f.write(f"{i}. {category}:\n")
        f.write(f"   成功率: {rate:.2%} ({category_success[category]}/{category_counts[category]})\n")
        
        if category_iou_values[category] or category_cd_values[category]:
            f.write(f"   平均IoU: {iou:.4f}\n")
            f.write(f"   平均CD: {cd:.4f}\n")
        else:
            f.write(f"   没有IoU/CD数据\n")
            
        f.write(f"   包含 {targets_in_category} 个目标物体\n")
        f.write(f"   目标列表: {', '.join(category_targets[category])}\n\n")
    
    # 添加总体统计
    total_attempts = sum(category_counts.values())
    total_success = sum(category_success.values())
    overall_rate = total_success / total_attempts if total_attempts > 0 else 0
    
    # 计算整体平均IoU和CD
    all_iou_values = [v for values in category_iou_values.values() for v in values]
    all_cd_values = [v for values in category_cd_values.values() for v in values]
    overall_iou = np.mean(all_iou_values) if all_iou_values else 0
    overall_cd = np.mean(all_cd_values) if all_cd_values else 0
    
    f.write("="*60 + "\n")
    f.write(f"总体统计：\n")
    f.write(f"总类别数: {len(category_counts)}\n")
    f.write(f"总目标物体数: {len(target_counts)}\n")
    f.write(f"总测试次数: {total_attempts}\n")
    f.write(f"总成功次数: {total_success}\n")
    f.write(f"总体成功率: {overall_rate:.2%}\n")
    
    if all_iou_values or all_cd_values:
        f.write(f"整体平均IoU: {overall_iou:.4f}\n")
        f.write(f"整体平均CD: {overall_cd:.4f}\n")
    else:
        f.write("没有IoU/CD数据\n")

# 创建类别成功率柱状图
plt.figure(figsize=(14, 8))

# 提取数据
categories = [cat for cat, _ in sorted_categories]
rates = [rate for _, rate in sorted_categories]
counts = [category_counts[cat] for cat, _ in sorted_categories]

# 创建柱状图
bars = plt.bar(categories, rates, color='skyblue')

# 添加计数标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'n={counts[i]}', ha='center', va='bottom', rotation=0)

# 添加标签和标题
plt.xlabel('类别')
plt.ylabel('成功率')
plt.title('不同类别的抓取成功率')
if rates:  # 检查rates列表是否为空
    plt.ylim(0, max(rates) + 0.1)  # 给标签留出空间
else:
    plt.ylim(0, 1.0)  # 如果rates为空，设置默认范围
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图表
plt.tight_layout()
plt.savefig(f"{output_dir}/category_success_rates.png", dpi=300)

# 创建类别样本数量饼图
plt.figure(figsize=(12, 10))

# 计算每个类别的样本百分比
category_percentages = {cat: (count/total_attempts)*100 for cat, count in category_counts.items()}
sorted_percentages = [(cat, category_percentages[cat]) for cat, _ in sorted_categories]

# 准备数据
pie_labels = [f"{cat} ({perc:.1f}%)" for cat, perc in sorted_percentages]
pie_sizes = [count for _, count in [(cat, category_counts[cat]) for cat, _ in sorted_categories]]

# 创建饼图
plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # 保持饼图为圆形
plt.title('各类别样本分布')

# 保存图表
plt.tight_layout()
plt.savefig(f"{output_dir}/category_distribution.png", dpi=300)

# 创建类别成功率与样本数量的散点图
plt.figure(figsize=(12, 8))

# 准备数据
x_vals = counts  # 样本数量
y_vals = rates   # 成功率
sizes = [len(category_targets[cat])*30 for cat, _ in sorted_categories]  # 气泡大小基于该类别的目标数量

# 创建散点图
scatter = plt.scatter(x_vals, y_vals, s=sizes, alpha=0.6, 
                      c=range(len(categories)), cmap='viridis')

# 添加类别标签
for i, cat in enumerate(categories):
    plt.annotate(cat, (x_vals[i], y_vals[i]), 
                 xytext=(5, 5), textcoords='offset points')

# 添加标签和标题
plt.xlabel('样本数量')
plt.ylabel('成功率')
plt.title('类别成功率与样本数量关系')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加趋势线
if len(x_vals) > 1:  # 确保有足够的数据点来拟合线
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    plt.plot(x_vals, p(x_vals), "r--", alpha=0.8)

# 保存图表
plt.tight_layout()
plt.savefig(f"{output_dir}/category_success_vs_samples.png", dpi=300)

# 新增: 仅当有IoU数据时创建IoU相关图表
if any(category_iou_values.values()):
    # 创建类别IoU柱状图
    plt.figure(figsize=(14, 8))
    iou_values = [category_avg_iou.get(cat, 0) for cat, _ in sorted_categories]
    bars = plt.bar(categories, iou_values, color='lightgreen')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{iou_values[i]:.3f}', ha='center', va='bottom', rotation=0)
                 
    plt.xlabel('类别')
    plt.ylabel('平均IoU')
    plt.title('不同类别的平均IoU')
    if iou_values:
        plt.ylim(0, max(iou_values) + 0.05)
    else:
        plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_iou.png", dpi=300)
    
    # 创建IoU与成功率散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(iou_values, rates, s=100, alpha=0.7, c=range(len(categories)), cmap='viridis')
    
    for i, cat in enumerate(categories):
        plt.annotate(cat, (iou_values[i], rates[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('平均IoU')
    plt.ylabel('成功率')
    plt.title('IoU与成功率关系')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if len(iou_values) > 1:
        z = np.polyfit(iou_values, rates, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(iou_values), max(iou_values), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/iou_vs_success.png", dpi=300)

# 新增: 仅当有CD数据时创建CD相关图表
if any(category_cd_values.values()):
    # 创建类别CD柱状图
    plt.figure(figsize=(14, 8))
    cd_values = [category_avg_cd.get(cat, 0) for cat, _ in sorted_categories]
    bars = plt.bar(categories, cd_values, color='salmon')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{cd_values[i]:.3f}', ha='center', va='bottom', rotation=0)
                 
    plt.xlabel('类别')
    plt.ylabel('平均Chamfer距离')
    plt.title('不同类别的平均Chamfer距离')
    if cd_values:
        plt.ylim(0, max(cd_values) + 0.01)
    else:
        plt.ylim(0, 0.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_cd.png", dpi=300)
    
    # 创建CD与成功率散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(cd_values, rates, s=100, alpha=0.7, c=range(len(categories)), cmap='viridis')
    
    for i, cat in enumerate(categories):
        plt.annotate(cat, (cd_values[i], rates[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('平均Chamfer距离')
    plt.ylabel('成功率')
    plt.title('Chamfer距离与成功率关系')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if len(cd_values) > 1:
        z = np.polyfit(cd_values, rates, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(cd_values), max(cd_values), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cd_vs_success.png", dpi=300)

print(f"分析完成。结果已保存到 {output_dir}")
print(f"类别总数: {len(category_counts)}")
print(f"表现最佳的类别: {sorted_categories[0][0]} 成功率: {sorted_categories[0][1]:.2%}")
print(f"表现最差的类别: {sorted_categories[-1][0]} 成功率: {sorted_categories[-1][1]:.2%}")

# 新增: 打印IoU和CD指标最佳的类别
if category_avg_iou and any(category_avg_iou.values()):
    best_iou_category = max(category_avg_iou.items(), key=lambda x: x[1])
    print(f"IoU最高的类别: {best_iou_category[0]} IoU: {best_iou_category[1]:.4f}")

if category_avg_cd and any(category_avg_cd.values()):
    best_cd_category = min(category_avg_cd.items(), key=lambda x: x[1])
    print(f"CD最低的类别: {best_cd_category[0]} CD: {best_cd_category[1]:.4f}")