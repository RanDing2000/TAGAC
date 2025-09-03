import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze evaluation results by category')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to evaluation results file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
    parser.add_argument('--data_type', type=str, choices=['ycb', 'acronym'], required=True, help='Dataset type (ycb or acronym)')
    return parser.parse_args()

def get_category_file(data_type):
    if data_type == 'ycb':
        return '/home/ran.ding/projects/TARGO/targo_eval_results/stastics_analysis/ycb_prompt_dict.json'
    elif data_type == 'acronym':
        return '/home/ran.ding/projects/TARGO/targo_eval_results/stastics_analysis/acronym_prompt_dict.json'
    else:
        raise ValueError(f'Unsupported data_type: {data_type}')

def load_category_mapping(category_file):
    with open(category_file, 'r') as f:
        return json.load(f)

def parse_eval_file(eval_file, category_mapping):
    target_counts = defaultdict(int)
    target_success = defaultdict(int)
    category_counts = defaultdict(int)
    category_success = defaultdict(int)
    category_targets = defaultdict(set)
    scene_target_map = {}

    with open(eval_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or 'Average' in line or 'Success' in line or 'Total' in line:
                continue

            parts = line.split('|') if '|' in line else line.split(', ')
            if len(parts) < 2:
                continue

            data_parts = parts[1].split(', ') if '|' in line else parts
            if len(data_parts) < 6:
                continue

            scene_id = data_parts[0].strip()
            target_name = data_parts[1].strip()
            success = int(data_parts[5].strip())

            scene_target_map[scene_id] = target_name
            target_counts[target_name] += 1
            target_success[target_name] += success

            category = category_mapping.get(target_name, 'Unknown')
            category_counts[category] += 1
            category_success[category] += success
            category_targets[category].add(target_name)

    return target_counts, target_success, category_counts, category_success, category_targets, scene_target_map

def extract_iou_cd(eval_file, scene_target_map, category_mapping):
    target_iou_values = defaultdict(list)
    target_cd_values = defaultdict(list)
    category_iou_values = defaultdict(list)
    category_cd_values = defaultdict(list)

    with open(eval_file, 'r') as f:
        content = f.read()

    for scene_id, target_name in scene_target_map.items():
        category = category_mapping.get(target_name, 'Unknown')
        pattern = rf"{scene_id},\s*{re.escape(target_name)},\s*[\d\.]+,\s*([\d\.]+),\s*([\d\.]+)"
        match = re.search(pattern, content)

        if match:
            iou = float(match.group(1))
            cd = float(match.group(2))
            target_iou_values[target_name].append(iou)
            target_cd_values[target_name].append(cd)
            category_iou_values[category].append(iou)
            category_cd_values[category].append(cd)

    return target_iou_values, target_cd_values, category_iou_values, category_cd_values

def calculate_statistics(target_counts, target_success, category_counts, category_success, category_iou_values, category_cd_values):
    target_success_rates = {t: target_success[t] / target_counts[t] for t in target_counts}
    category_success_rates = {c: category_success[c] / category_counts[c] for c in category_counts}
    category_avg_iou = {c: np.mean(category_iou_values[c]) if category_iou_values[c] else 0 for c in category_counts}
    category_avg_cd = {c: np.mean(category_cd_values[c]) if category_cd_values[c] else 0 for c in category_counts}
    return target_success_rates, category_success_rates, category_avg_iou, category_avg_cd

def save_statistics(output_dir, category_success_rates, category_avg_iou, category_avg_cd, category_counts, category_success, category_targets):
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/category_success_rates.json", 'w') as f:
        json.dump({
            "success_rates": category_success_rates,
            "avg_iou": category_avg_iou,
            "avg_cd": category_avg_cd
        }, f, indent=4)

    sorted_categories = sorted(category_success_rates.items(), key=lambda x: x[1], reverse=True)
    total_attempts = sum(category_counts.values())
    total_success = sum(category_success.values())
    overall_rate = total_success / total_attempts if total_attempts else 0

    with open(f"{output_dir}/category_success_summary.txt", 'w') as f:
        f.write("Category Success Rate and Quality Statistics:\n")
        f.write("="*60 + "\n\n")
        for i, (category, rate) in enumerate(sorted_categories, 1):
            targets_in_category = len(category_targets[category])
            iou = category_avg_iou.get(category, 0)
            cd = category_avg_cd.get(category, 0)
            f.write(f"{i}. {category}:\n")
            f.write(f"   Success Rate: {rate:.2%} ({category_success[category]}/{category_counts[category]})\n")
            f.write(f"   Average IoU: {iou:.4f}\n   Average CD: {cd:.4f}\n")
            f.write(f"   Targets: {', '.join(category_targets[category])}\n\n")

        f.write("="*60 + "\n")
        f.write(f"Total categories: {len(category_counts)}\n")
        f.write(f"Total test attempts: {total_attempts}\n")
        f.write(f"Total successful attempts: {total_success}\n")
        f.write(f"Overall success rate: {overall_rate:.2%}\n")

def plot_statistics(output_dir, category_success_rates, category_avg_iou, category_avg_cd, category_counts, category_targets):
    sorted_categories = sorted(category_success_rates.items(), key=lambda x: x[1], reverse=True)
    categories = [cat for cat, _ in sorted_categories]
    rates = [rate for _, rate in sorted_categories]
    counts = [category_counts[cat] for cat in categories]

    # Success rate bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(categories, rates)
    plt.xlabel('Category')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_success_rates.png", dpi=300)
    plt.close()

    # Sample count pie chart
    plt.figure(figsize=(12, 10))
    plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Category Sample Distribution')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_distribution.png", dpi=300)
    plt.close()

def main():
    args = parse_arguments()
    category_file = get_category_file(args.data_type)
    category_mapping = load_category_mapping(category_file)

    target_counts, target_success, category_counts, category_success, category_targets, scene_target_map = parse_eval_file(args.eval_file, category_mapping)
    target_iou_values, target_cd_values, category_iou_values, category_cd_values = extract_iou_cd(args.eval_file, scene_target_map, category_mapping)
    target_success_rates, category_success_rates, category_avg_iou, category_avg_cd = calculate_statistics(
        target_counts, target_success, category_counts, category_success, category_iou_values, category_cd_values)

    save_statistics(args.output_dir, category_success_rates, category_avg_iou, category_avg_cd, category_counts, category_success, category_targets)
    plot_statistics(args.output_dir, category_success_rates, category_avg_iou, category_avg_cd, category_counts, category_targets)
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()