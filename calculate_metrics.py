import os
import json
from pathlib import Path
import numpy as np

def calculate_average_metrics(base_dir):
    """
    Calculate average IoU and Chamfer L1 distance from all scene_metadata.txt files
    
    Args:
        base_dir (str): Base directory containing scene folders
    """
    # Initialize lists to store metrics
    iou_values = []
    cd_values = []
    
    # Get all subdirectories
    base_path = Path(base_dir)
    
    try:
        # Iterate through all subdirectories
        for scene_dir in base_path.iterdir():
            if not scene_dir.is_dir():
                continue
                
            metadata_file = scene_dir / "scene_metadata.txt"
            if not metadata_file.exists():
                continue
                
            # Read and parse metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.loads(f.read())
                    
                # Extract metrics
                if 'iou' in metadata and 'cd' in metadata:
                    iou_values.append(metadata['iou'])
                    cd_values.append(metadata['cd'])
            except json.JSONDecodeError as e:
                print(f"Error parsing {metadata_file}: {e}")
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")
        
        # Calculate averages
        if iou_values and cd_values:
            avg_iou = np.mean(iou_values)
            avg_cd = np.mean(cd_values)
            
            # Calculate standard deviations
            std_iou = np.std(iou_values)
            std_cd = np.std(cd_values)
            
            print(f"\n总场景数: {len(iou_values)}")
            print(f"\n平均指标:")
            print(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}")
            print(f"Average Chamfer L1: {avg_cd:.4f} ± {std_cd:.4f}")
            
            # Save results to a summary file
            summary_path = os.path.join(base_dir, "metrics_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Total scenes: {len(iou_values)}\n")
                f.write(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}\n")
                f.write(f"Average Chamfer L1: {avg_cd:.4f} ± {std_cd:.4f}\n")
            
            return avg_iou, avg_cd, len(iou_values)
        else:
            print("No valid metrics found in any scene_metadata.txt files")
            return None, None, 0
            
    except Exception as e:
        print(f"Error processing directory {base_dir}: {e}")
        return None, None, 0

if __name__ == "__main__":
    base_dir = "/usr/stud/dira/GraspInClutter/targo/eval_results_test/targo/2025-01-29_09-22-41"
    avg_iou, avg_cd, total_scenes = calculate_average_metrics(base_dir) 