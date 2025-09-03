import os
import json
from pathlib import Path
import numpy as np
import trimesh
import sys

# Add the src directory to the path to import utils_giga
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils_giga import compute_multi_object_completion_metrics_messy_kitchen

def load_meshes_from_directory(scene_dir):
    """
    Load predicted and ground truth meshes from a scene directory
    
    Args:
        scene_dir (Path): Path to scene directory
        
    Returns:
        tuple: (pred_meshes, gt_meshes) - lists of trimesh.Trimesh objects
    """
    pred_meshes = []
    gt_meshes = []
    
    try:
        # Load predicted meshes (aligned_pred_obj_*.ply)
        pred_files = sorted(scene_dir.glob("aligned_pred_obj_*.ply"))
        for pred_file in pred_files:
            try:
                mesh = trimesh.load_mesh(str(pred_file))
                ## check watertight
                if not mesh.is_watertight:
                    print(f"  Warning: Predicted mesh {pred_file.name} is not watertight")
                    mesh.fill_holes()
                    mesh.remove_degenerate_faces()
                    mesh.remove_duplicate_faces()
                    mesh.remove_infinite_values()
                    mesh.remove_unreferenced_vertices()
                pred_meshes.append(mesh)
                print(f"  Loaded predicted mesh: {pred_file.name}")
            except Exception as e:
                print(f"  Error loading {pred_file.name}: {e}")
        
        # Load ground truth meshes (gt_object_*.ply)
        gt_files = sorted(scene_dir.glob("gt_object_*.ply"))
        for gt_file in gt_files:
            try:
                mesh = trimesh.load_mesh(str(gt_file))
                gt_meshes.append(mesh)
                print(f"  Loaded GT mesh: {gt_file.name}")
            except Exception as e:
                print(f"  Error loading {gt_file.name}: {e}")
        
        print(f"  Scene {scene_dir.name}: {len(pred_meshes)} pred meshes, {len(gt_meshes)} GT meshes")
        
    except Exception as e:
        print(f"  Error processing scene directory {scene_dir}: {e}")
    
    return pred_meshes, gt_meshes

def evaluate_messy_kitchen_scenes(base_dir):
    """
    Evaluate all messy kitchen scenes in the our_model directory
    
    Args:
        base_dir (str): Base directory containing our_model folder
    """
    base_path = Path(base_dir)
    our_model_path = base_path / "our_model"
    
    if not our_model_path.exists():
        print(f"Error: our_model directory not found at {our_model_path}")
        return
    
    print(f"Processing messy kitchen results from: {our_model_path}")
    
    # Initialize lists to store metrics
    all_scene_metrics = []
    scene_names = []
    
    # Get all subdirectories in our_model
    scene_dirs = [d for d in our_model_path.iterdir() if d.is_dir()]
    print(f"Found {len(scene_dirs)} scene directories")
    
    for scene_dir in scene_dirs:
        print(f"\nProcessing scene: {scene_dir.name}")
        
        # Load meshes for this scene
        pred_meshes, gt_meshes = load_meshes_from_directory(scene_dir)
        
        if not pred_meshes or not gt_meshes:
            print(f"  Skipping scene {scene_dir.name}: No valid meshes found")
            continue
        
        try:
            # Call the messy kitchen evaluation function
            metrics = compute_multi_object_completion_metrics_messy_kitchen(
                gt_meshes=gt_meshes,
                pred_meshes=pred_meshes
            )
            
            # Store metrics
            all_scene_metrics.append(metrics)
            scene_names.append(scene_dir.name)
            
            print(f"  Scene {scene_dir.name} metrics:")
            print(f"    CD: {metrics['chamfer_distance']:.6f} ± {metrics['chamfer_distance_std']:.6f}")
            print(f"    IoU: {metrics['iou']:.6f} ± {metrics['iou_std']:.6f}")
            
        except Exception as e:
            print(f"  Error evaluating scene {scene_dir.name}: {e}")
            continue
    
    # Calculate overall statistics
    if all_scene_metrics:
        calculate_overall_statistics(all_scene_metrics, scene_names, our_model_path)
    else:
        print("No valid scene metrics found")

def calculate_overall_statistics(all_scene_metrics, scene_names, output_dir):
    """
    Calculate and display overall statistics from all scene metrics
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        scene_names (list): List of scene names
        output_dir (Path): Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*60}")
    
    # Extract metrics
    cds = [m['chamfer_distance'] for m in all_scene_metrics if m['chamfer_distance'] != float('inf')]
    ious = [m['iou'] for m in all_scene_metrics if m['iou'] > 0]
    
    # Calculate statistics
    if cds:
        avg_cd = np.mean(cds)
        std_cd = np.std(cds)
        min_cd = np.min(cds)
        max_cd = np.max(cds)
        print(f"Chamfer Distance:")
        print(f"  Average: {avg_cd:.6f} ± {std_cd:.6f}")
        print(f"  Range: [{min_cd:.6f}, {max_cd:.6f}]")
        print(f"  Valid scenes: {len(cds)}/{len(all_scene_metrics)}")
    else:
        print("Chamfer Distance: No valid values")
    
    if ious:
        avg_iou = np.mean(ious)
        std_iou = np.std(ious)
        min_iou = np.min(ious)
        max_iou = np.max(ious)
        print(f"\nIoU:")
        print(f"  Average: {avg_iou:.6f} ± {std_iou:.6f}")
        print(f"  Range: [{min_iou:.6f}, {max_iou:.6f}]")
        print(f"  Valid scenes: {len(ious)}/{len(all_scene_metrics)}")
    else:
        print("\nIoU: No valid values")
    
    # Save detailed results
    save_detailed_results(all_scene_metrics, scene_names, output_dir)
    
    # Save summary
    save_summary_results(all_scene_metrics, output_dir)

def save_detailed_results(all_scene_metrics, scene_names, output_dir):
    """
    Save detailed results for each scene
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        scene_names (list): List of scene names
        output_dir (Path): Directory to save results
    """
    detailed_file = output_dir / "detailed_metrics.txt"
    
    with open(detailed_file, 'w') as f:
        f.write("Detailed Metrics for Each Scene\n")
        f.write("="*50 + "\n\n")
        
        for i, (metrics, scene_name) in enumerate(zip(all_scene_metrics, scene_names)):
            f.write(f"Scene {i+1}: {scene_name}\n")
            f.write(f"  Chamfer Distance: {metrics['chamfer_distance']:.6f} ± {metrics['chamfer_distance_std']:.6f}\n")
            f.write(f"  IoU: {metrics['iou']:.6f} ± {metrics['iou_std']:.6f}\n")
            
            if 'per_object_cds' in metrics and metrics['per_object_cds']:
                f.write(f"  Per-object CDs: {[f'{cd:.6f}' for cd in metrics['per_object_cds']]}\n")
            if 'per_object_ious' in metrics and metrics['per_object_ious']:
                f.write(f"  Per-object IoUs: {[f'{iou:.6f}' for iou in metrics['per_object_ious']]}\n")
            
            f.write("\n")
    
    print(f"\nDetailed results saved to: {detailed_file}")

def save_summary_results(all_scene_metrics, output_dir):
    """
    Save summary results
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        output_dir (Path): Directory to save results
    """
    summary_file = output_dir / "metrics_summary.txt"
    
    # Calculate overall statistics
    cds = [m['chamfer_distance'] for m in all_scene_metrics if m['chamfer_distance'] != float('inf')]
    ious = [m['iou'] for m in all_scene_metrics if m['iou'] > 0]
    
    with open(summary_file, 'w') as f:
        f.write("Messy Kitchen Evaluation Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total scenes processed: {len(all_scene_metrics)}\n")
        f.write(f"Scenes with valid CD: {len(cds)}\n")
        f.write(f"Scenes with valid IoU: {len(ious)}\n\n")
        
        if cds:
            f.write(f"Chamfer Distance:\n")
            f.write(f"  Average: {np.mean(cds):.6f} ± {np.std(cds):.6f}\n")
            f.write(f"  Min: {np.min(cds):.6f}\n")
            f.write(f"  Max: {np.max(cds):.6f}\n\n")
        
        if ious:
            f.write(f"IoU:\n")
            f.write(f"  Average: {np.mean(ious):.6f} ± {np.std(ious):.6f}\n")
            f.write(f"  Min: {np.min(ious):.6f}\n")
            f.write(f"  Max: {np.max(ious):.6f}\n")
    
    print(f"Summary results saved to: {summary_file}")

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
    # Path to messy kitchen results
    base_dir = "/home/ran.ding/projects/TARGO/messy_kitchen_results"
    
    print("Messy Kitchen Scene Evaluation")
    print("="*50)
    
    # Evaluate messy kitchen scenes
    evaluate_messy_kitchen_scenes(base_dir)
    
    print(f"\nEvaluation completed for: {base_dir}") 