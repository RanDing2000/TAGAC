import os
import json
import glob

def analyze_metadata_files(directory):
    # Initialize variables to track best results
    best_cd = float('inf')
    best_cd_scene = ''
    best_iou = float('-inf')
    best_iou_scene = ''
    
    # Find all scene_metadata.txt files
    pattern = os.path.join(directory, "**/scene_metadata.txt")
    for filepath in glob.glob(pattern, recursive=True):
        with open(filepath, 'r') as f:
            data = json.loads(f.read())
            
            # Track best CD (lower is better)
            if data['cd'] < best_cd:
                best_cd = data['cd']
                best_cd_scene = data['scene_id']
                
            # Track best IOU (higher is better)
            if data['iou'] > best_iou:
                best_iou = data['iou']
                best_iou_scene = data['scene_id']
    
    return {
        'best_cd': {'value': best_cd, 'scene': best_cd_scene},
        'best_iou': {'value': best_iou, 'scene': best_iou_scene}
    }

# 分析目录
directory = "/usr/stud/dira/GraspInClutter/targo/eval_results_test/targo/2025-01-29_09-22-41"
results = analyze_metadata_files(directory)

print(f"最佳CD (最小值):")
print(f"场景: {results['best_cd']['scene']}")
print(f"CD值: {results['best_cd']['value']}")
print(f"\n最佳IoU (最大值):")
print(f"场景: {results['best_iou']['scene']}")
print(f"IoU值: {results['best_iou']['value']}")