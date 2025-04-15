import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze evaluation results by category')
parser.add_argument('--eval_file', type=str, required=True, 
                    help='Path to evaluation results file')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory to save output results')
parser.add_argument('--category_file', type=str, 
                    default="/usr/stud/dira/GraspInClutter/targo/targo_eval_results/stastics_analysis/ycb_prompt_dict.json",
                    help='Path to category mapping file')
args = parser.parse_args()

# Path settings
eval_file = args.eval_file
category_file = args.category_file
output_dir = args.output_dir

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load category mapping
with open(category_file, 'r') as f:
    category_mapping = json.load(f)

# Initialize data structures
target_counts = defaultdict(int)
target_success = defaultdict(int)
category_counts = defaultdict(int)
category_success = defaultdict(int)
category_targets = defaultdict(set)  # Record targets in each category

# New: IoU and CD statistics data structures
target_iou_values = defaultdict(list)
target_cd_values = defaultdict(list)
category_iou_values = defaultdict(list)
category_cd_values = defaultdict(list)

# Mapping from scene ID to target name
scene_target_map = {}

# New: Find and read meta evaluation files containing CD and IoU
def find_meta_evaluation_files(base_dir):
    print(f"Searching directory: {base_dir}")
    meta_files = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in ["meta_evaluation.txt", "meta_evaluations.txt"]:
                print(f"Found file: {os.path.join(root, file)}")
                try:
                    # Try to extract scene ID from path
                    scene_id = os.path.basename(os.path.dirname(os.path.dirname(root)))
                    meta_files[scene_id] = os.path.join(root, file)
                except:
                    # If path structure doesn't match expectations, use file path as key
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    meta_files[rel_path] = os.path.join(root, file)
    return meta_files

# Parse evaluation file, get scene ID to target name mapping and calculate success rate
print("Parsing evaluation file...")
with open(eval_file, 'r') as f:
    for line in f:
        line = line.strip()
        # Skip empty lines and summary lines
        if not line or 'Average' in line or 'Success' in line or 'Total' in line:
            continue
        
        # Parse line
        if '|' in line:
            # New format: ID| Scene ID, Object ID, Value1, Value2, Value3, Success flag(0/1)
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
            # Old format: Scene ID, Object ID, Value1, Value2, Value3, Success flag(0/1)
            parts = line.split(', ')
            if len(parts) < 6:
                print(f"Skipping line with unexpected format: {line}")
                continue
            
            scene_id = parts[0]
            target_name = parts[1]
            success = int(parts[5])
        
        try:
            # Store scene ID to target name mapping
            scene_target_map[scene_id] = target_name
            
            # Update target counts
            target_counts[target_name] += 1
            target_success[target_name] += success
            
            # Get category and update category counts
            category = category_mapping.get(target_name, "Unknown")
            category_counts[category] += 1
            category_success[category] += success
            category_targets[category].add(target_name)
            
        except (ValueError, IndexError) as e:
            print(f"Error processing line: {line}")
            print(f"Error details: {e}")
            continue

# New: Try to find and process meta evaluation files containing CD and IoU
# meta_evaluation_files = find_meta_evaluation_files("eval_results_train_full-middle-occlusion-1000")
print("Extracting CD and IoU values...")
try:
    with open(eval_file, 'r') as f:
        content = f.read()
        
        # Iterate through processed scenes and targets
        for scene_id, target_name in scene_target_map.items():
            category = category_mapping.get(target_name, "Unknown")
            
            # Extract CD and IoU values based on meta evaluation file format
            # Reference format: "Scene_ID, Target_Name, Occlusion_Level, IoU, CD, Success"
            pattern = rf"{scene_id},\s*{target_name},\s*[\d\.]+,\s*([\d\.]+),\s*([\d\.]+)"
            match = re.search(pattern, content)
            
            if match:
                iou = float(match.group(1))
                cd = float(match.group(2))
                
                target_cd_values[target_name].append(cd)
                target_iou_values[target_name].append(iou)
                category_cd_values[category].append(cd)
                category_iou_values[category].append(iou)
                print(f"Extracted {target_name} (Category: {category}) CD={cd}, IoU={iou}")
except Exception as e:
    print(f"Error processing file {eval_file}: {e}")

# Calculate success rate for each target
target_success_rates = {}
for target in target_counts:
    if target_counts[target] > 0:
        success_rate = target_success[target] / target_counts[target]
        target_success_rates[target] = success_rate

# Calculate success rate and average IoU/CD for each category
category_success_rates = {}
category_avg_iou = {}
category_avg_cd = {}

for category in category_counts:
    if category_counts[category] > 0:
        success_rate = category_success[category] / category_counts[category]
        category_success_rates[category] = success_rate
    
    # Calculate average IoU and CD
    if category_iou_values[category]:
        category_avg_iou[category] = np.mean(category_iou_values[category])
    else:
        category_avg_iou[category] = 0
        
    if category_cd_values[category]:
        category_avg_cd[category] = np.mean(category_cd_values[category])
    else:
        category_avg_cd[category] = 0

# Sort categories by success rate
sorted_categories = sorted(category_success_rates.items(), key=lambda x: x[1], reverse=True)

# Generate category report
with open(f"{output_dir}/category_success_rates.json", 'w') as f:
    data = {
        "success_rates": category_success_rates,
        "avg_iou": category_avg_iou,
        "avg_cd": category_avg_cd
    }
    json.dump(data, f, indent=4)

with open(f"{output_dir}/category_success_summary.txt", 'w') as f:
    f.write("Category Success Rate and Quality Statistics:\n")
    f.write("="*60 + "\n\n")
    
    for i, (category, rate) in enumerate(sorted_categories, 1):
        targets_in_category = len(category_targets[category])
        iou = category_avg_iou.get(category, 0)
        cd = category_avg_cd.get(category, 0)
        
        f.write(f"{i}. {category}:\n")
        f.write(f"   Success Rate: {rate:.2%} ({category_success[category]}/{category_counts[category]})\n")
        
        if category_iou_values[category] or category_cd_values[category]:
            f.write(f"   Average IoU: {iou:.4f}\n")
            f.write(f"   Average CD: {cd:.4f}\n")
        else:
            f.write(f"   No IoU/CD data\n")
            
        f.write(f"   Contains {targets_in_category} target objects\n")
        f.write(f"   Target list: {', '.join(category_targets[category])}\n\n")
    
    # Add overall statistics
    total_attempts = sum(category_counts.values())
    total_success = sum(category_success.values())
    overall_rate = total_success / total_attempts if total_attempts > 0 else 0
    
    # Calculate overall average IoU and CD
    all_iou_values = [v for values in category_iou_values.values() for v in values]
    all_cd_values = [v for values in category_cd_values.values() for v in values]
    overall_iou = np.mean(all_iou_values) if all_iou_values else 0
    overall_cd = np.mean(all_cd_values) if all_cd_values else 0
    
    f.write("="*60 + "\n")
    f.write(f"Overall Statistics:\n")
    f.write(f"Total categories: {len(category_counts)}\n")
    f.write(f"Total target objects: {len(target_counts)}\n")
    f.write(f"Total test attempts: {total_attempts}\n")
    f.write(f"Total successful attempts: {total_success}\n")
    f.write(f"Overall success rate: {overall_rate:.2%}\n")
    
    if all_iou_values or all_cd_values:
        f.write(f"Overall average IoU: {overall_iou:.4f}\n")
        f.write(f"Overall average CD: {overall_cd:.4f}\n")
    else:
        f.write("No IoU/CD data\n")

# Create category success rate bar chart
plt.figure(figsize=(14, 8))

# Extract data
categories = [cat for cat, _ in sorted_categories]
rates = [rate for _, rate in sorted_categories]
counts = [category_counts[cat] for cat, _ in sorted_categories]

# Create bar chart
bars = plt.bar(categories, rates, color='skyblue')

# Add count labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'n={counts[i]}', ha='center', va='bottom', rotation=0)

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Success Rate')
plt.title('Grasp Success Rate by Category')
if rates:  # Check if rates list is empty
    plt.ylim(0, max(rates) + 0.1)  # Leave space for labels
else:
    plt.ylim(0, 1.0)  # Set default range if rates is empty
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save chart
plt.tight_layout()
plt.savefig(f"{output_dir}/category_success_rates.png", dpi=300)

# Create category sample count pie chart
plt.figure(figsize=(12, 10))

# Calculate percentage of samples for each category
category_percentages = {cat: (count/total_attempts)*100 for cat, count in category_counts.items()}
sorted_percentages = [(cat, category_percentages[cat]) for cat, _ in sorted_categories]

# Prepare data
pie_labels = [f"{cat} ({perc:.1f}%)" for cat, perc in sorted_percentages]
pie_sizes = [count for _, count in [(cat, category_counts[cat]) for cat, _ in sorted_categories]]

# Create pie chart
plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Keep pie chart circular
plt.title('Category Sample Distribution')

# Save chart
plt.tight_layout()
plt.savefig(f"{output_dir}/category_distribution.png", dpi=300)

# Create scatter plot of category success rate vs sample count
plt.figure(figsize=(12, 8))

# Prepare data
x_vals = counts  # Sample counts
y_vals = rates   # Success rates
sizes = [len(category_targets[cat])*30 for cat, _ in sorted_categories]  # Bubble size based on number of targets in category

# Create scatter plot
scatter = plt.scatter(x_vals, y_vals, s=sizes, alpha=0.6, 
                      c=range(len(categories)), cmap='viridis')

# Add category labels
for i, cat in enumerate(categories):
    plt.annotate(cat, (x_vals[i], y_vals[i]), 
                 xytext=(5, 5), textcoords='offset points')

# Add labels and title
plt.xlabel('Sample Count')
plt.ylabel('Success Rate')
plt.title('Category Success Rate vs Sample Count')
plt.grid(True, linestyle='--', alpha=0.7)

# Add trend line
if len(x_vals) > 1:  # Ensure there are enough data points to fit a line
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    plt.plot(x_vals, p(x_vals), "r--", alpha=0.8)

# Save chart
plt.tight_layout()
plt.savefig(f"{output_dir}/category_success_vs_samples.png", dpi=300)

# New: Create IoU-related charts only if IoU data exists
if any(category_iou_values.values()):
    # Create category IoU bar chart
    plt.figure(figsize=(14, 8))
    iou_values = [category_avg_iou.get(cat, 0) for cat, _ in sorted_categories]
    bars = plt.bar(categories, iou_values, color='lightgreen')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{iou_values[i]:.3f}', ha='center', va='bottom', rotation=0)
                 
    plt.xlabel('Category')
    plt.ylabel('Average IoU')
    plt.title('Average IoU by Category')
    if iou_values:
        plt.ylim(0, max(iou_values) + 0.05)
    else:
        plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_iou.png", dpi=300)
    
    # Create IoU vs success rate scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(iou_values, rates, s=100, alpha=0.7, c=range(len(categories)), cmap='viridis')
    
    for i, cat in enumerate(categories):
        plt.annotate(cat, (iou_values[i], rates[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Average IoU')
    plt.ylabel('Success Rate')
    plt.title('IoU vs Success Rate Relationship')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if len(iou_values) > 1:
        z = np.polyfit(iou_values, rates, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(iou_values), max(iou_values), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/iou_vs_success.png", dpi=300)

# New: Create CD-related charts only if CD data exists
if any(category_cd_values.values()):
    # Create category CD bar chart
    plt.figure(figsize=(14, 8))
    cd_values = [category_avg_cd.get(cat, 0) for cat, _ in sorted_categories]
    bars = plt.bar(categories, cd_values, color='salmon')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{cd_values[i]:.3f}', ha='center', va='bottom', rotation=0)
                 
    plt.xlabel('Category')
    plt.ylabel('Average Chamfer Distance')
    plt.title('Average Chamfer Distance by Category')
    if cd_values:
        plt.ylim(0, max(cd_values) + 0.01)
    else:
        plt.ylim(0, 0.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_cd.png", dpi=300)
    
    # Create CD vs success rate scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(cd_values, rates, s=100, alpha=0.7, c=range(len(categories)), cmap='viridis')
    
    for i, cat in enumerate(categories):
        plt.annotate(cat, (cd_values[i], rates[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Average Chamfer Distance')
    plt.ylabel('Success Rate')
    plt.title('Chamfer Distance vs Success Rate Relationship')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if len(cd_values) > 1:
        z = np.polyfit(cd_values, rates, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(cd_values), max(cd_values), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cd_vs_success.png", dpi=300)

print(f"Analysis complete. Results saved to {output_dir}")
print(f"Total categories: {len(category_counts)}")
print(f"Best performing category: {sorted_categories[0][0]} Success rate: {sorted_categories[0][1]:.2%}")
print(f"Worst performing category: {sorted_categories[-1][0]} Success rate: {sorted_categories[-1][1]:.2%}")

# New: Print categories with best IoU and CD metrics
if category_avg_iou and any(category_avg_iou.values()):
    best_iou_category = max(category_avg_iou.items(), key=lambda x: x[1])
    print(f"Category with highest IoU: {best_iou_category[0]} IoU: {best_iou_category[1]:.4f}")

if category_avg_cd and any(category_avg_cd.values()):
    best_cd_category = min(category_avg_cd.items(), key=lambda x: x[1])
    print(f"Category with lowest CD: {best_cd_category[0]} CD: {best_cd_category[1]:.4f}")