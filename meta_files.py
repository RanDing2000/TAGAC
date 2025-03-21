import os
import json
import glob

def analyze_metadata_files(directory):
    # List to store all CD values and their corresponding scenes
    cd_list = []
    
    # Find all scene_metadata.txt files
    pattern = os.path.join(directory, "**/scene_metadata.txt")
    for filepath in glob.glob(pattern, recursive=True):
        with open(filepath, 'r') as f:
            data = json.loads(f.read())
            
            # Append CD and scene_id to the list
            cd_list.append((data['cd'], data['scene_id']))
    
    # Sort the list by CD value in descending order
    cd_list.sort(key=lambda x: x[0], reverse=True)
    
    # Get the highest 50 CD values
    highest_50_cd = cd_list[:50]
    
    return highest_50_cd

# 分析目录
directory = "/usr/stud/dira/GraspInClutter/targo/eval_results_test/targo/2025-01-29_09-22-41"
highest_50_cd = analyze_metadata_files(directory)

print("CD最高的50个场景:")
for i, (cd, scene) in enumerate(highest_50_cd, start=1):
    print(f"{i}. 场景: {scene}, CD值: {cd}")