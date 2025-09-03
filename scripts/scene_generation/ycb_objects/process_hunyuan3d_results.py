#!/usr/bin/env python3
"""
Script to process Hunyuan3D reconstruction results for YCB dataset.
Reads PLY files from reconstruction directories, generates point clouds and TSDF,
and saves them as .npy files in corresponding scenes directories.
"""

import os
import sys
import numpy as np
import trimesh
from tqdm import tqdm
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.utils_giga import mesh_to_tsdf

def process_hunyuan3d_results(hunyuan_results_path, max_scenes=None, dry_run=False):
    """
    Process Hunyuan3D reconstruction results to generate point clouds and TSDF.
    
    Args:
        hunyuan_results_path: Path to Hunyuan3D results directory
        max_scenes: Maximum number of scenes to process (None for all)
        dry_run: If True, only analyze without generating files
    """
    if not os.path.exists(hunyuan_results_path):
        print(f"Error: Hunyuan3D results directory not found: {hunyuan_results_path}")
        return
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(hunyuan_results_path) 
                  if os.path.isdir(os.path.join(hunyuan_results_path, d)) 
                  and not d.startswith('.')]
    scene_dirs.sort()
    
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]
    
    print(f"Found {len(scene_dirs)} scene directories to process")
    
    # Statistics
    stats = {
        'total_dirs': len(scene_dirs),
        'processed': 0,
        'already_has_files': 0,
        'missing_ply': 0,
        'errors': 0,
        'error_dirs': []
    }
    
    for scene_dir in tqdm(scene_dirs, desc="Processing Hunyuan3D results"):
        scene_path = os.path.join(hunyuan_results_path, scene_dir)
        
        try:
            # Define paths
            reconstruction_dir = os.path.join(scene_path, 'reconstruction')
            scenes_dir = os.path.join(scene_path, 'scenes')
            ply_file = os.path.join(reconstruction_dir, 'targ_obj_hy3dgen_align.ply')
            
            pc_output_file = os.path.join(scenes_dir, 'complete_targ_hunyuan_pc.npy')
            tsdf_output_file = os.path.join(scenes_dir, 'complete_targ_hunyuan_tsdf.npy')
            
            # Check if output files already exist
            if os.path.exists(pc_output_file) and os.path.exists(tsdf_output_file):
                stats['already_has_files'] += 1
                continue
            
            # Check if PLY file exists
            if not os.path.exists(ply_file):
                stats['missing_ply'] += 1
                print(f"Warning: Missing PLY file in {scene_dir}")
                continue
            
            # Create scenes directory if it doesn't exist
            os.makedirs(scenes_dir, exist_ok=True)
            
            if dry_run:
                print(f"Would process {scene_dir}: PLY found, would generate PC and TSDF")
                stats['processed'] += 1
                continue
            
            # Load PLY file
            try:
                mesh = trimesh.load(ply_file)
                
                # Handle the case where trimesh.load returns a Scene object
                if isinstance(mesh, trimesh.Scene):
                    # Extract the first mesh from the scene
                    if len(mesh.geometry) > 0:
                        mesh = list(mesh.geometry.values())[0]
                    else:
                        print(f"Warning: Empty scene in PLY file for {scene_dir}")
                        stats['errors'] += 1
                        stats['error_dirs'].append(scene_dir)
                        continue
                
                # Apply basic mesh fixes
                mesh.fix_normals()
                
                # Use new methods to clean up mesh
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.update_faces(mesh.unique_faces())
                mesh.remove_unreferenced_vertices()
                
                # Basic validation - check if mesh has vertices and faces after cleanup
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    print(f"Warning: Mesh became empty after cleanup in {scene_dir}")
                    stats['errors'] += 1
                    stats['error_dirs'].append(scene_dir)
                    continue
                    
            except Exception as e:
                print(f"Error loading PLY file for {scene_dir}: {e}")
                stats['errors'] += 1
                stats['error_dirs'].append(scene_dir)
                continue
            
            # Generate point cloud (sample 4096 points)
            try:
                point_cloud = mesh.sample(4096)
                
                # Validate point cloud
                if point_cloud is None or len(point_cloud) == 0:
                    print(f"Error: Empty point cloud generated for {scene_dir}")
                    stats['errors'] += 1
                    stats['error_dirs'].append(scene_dir)
                    continue
                
                # Save point cloud
                np.save(pc_output_file, point_cloud.astype(np.float32))
                
            except Exception as e:
                print(f"Error generating point cloud for {scene_dir}: {e}")
                stats['errors'] += 1
                stats['error_dirs'].append(scene_dir)
                continue
            
            # Generate TSDF using mesh_to_tsdf function
            try:
                complete_target_tsdf = mesh_to_tsdf(mesh)
                
                # Validate TSDF
                if complete_target_tsdf is None or complete_target_tsdf.size == 0:
                    print(f"Error: Empty TSDF generated for {scene_dir}")
                    stats['errors'] += 1
                    stats['error_dirs'].append(scene_dir)
                    continue
                
                # Save TSDF
                np.save(tsdf_output_file, complete_target_tsdf)
                stats['processed'] += 1
                
            except Exception as e:
                print(f"Error generating TSDF for {scene_dir}: {e}")
                stats['errors'] += 1
                stats['error_dirs'].append(scene_dir)
                continue
                
        except Exception as e:
            print(f"Error processing {scene_dir}: {e}")
            stats['errors'] += 1
            stats['error_dirs'].append(scene_dir)
            continue
    
    # Print statistics
    print("\n" + "="*50)
    print("PROCESSING STATISTICS")
    print("="*50)
    print(f"Total directories: {stats['total_dirs']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Already had files: {stats['already_has_files']}")
    print(f"Missing PLY files: {stats['missing_ply']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['error_dirs']:
        print(f"\nError directories ({len(stats['error_dirs'])}):")
        for error_dir in stats['error_dirs'][:10]:  # Show first 10
            print(f"  - {error_dir}")
        if len(stats['error_dirs']) > 10:
            print(f"  ... and {len(stats['error_dirs']) - 10} more")
    
    if stats['total_dirs'] - stats['already_has_files'] > 0:
        print(f"\nSuccess rate: {stats['processed']/(stats['total_dirs'] - stats['already_has_files'])*100:.1f}%")
    
    # Save error directories to txt file if there are any errors
    if stats['error_dirs']:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_error_dirs_to_txt(stats['error_dirs'], script_dir, "hunyuan3d")

def save_error_dirs_to_txt(error_dirs, script_dir, dataset_name="hunyuan3d"):
    """
    Save error directory names to a txt file with timestamp.
    
    Args:
        error_dirs: List of error directory names
        script_dir: Directory where the script is located
        dataset_name: Name of the dataset for filename
    """
    if not error_dirs:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_failed_dirs_{timestamp}.txt"
    filepath = os.path.join(script_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Failed directories for {dataset_name.upper()}\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total failed directories: {len(error_dirs)}\n")
            f.write("#" + "="*50 + "\n\n")
            
            for error_dir in error_dirs:
                f.write(f"{error_dir}\n")
        
        print(f"Error directories list saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Warning: Failed to save error directories list: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Process Hunyuan3D reconstruction results to generate point clouds and TSDF')
    parser.add_argument('--hunyuan_path', type=str, 
                       default='/home/ran.ding/projects/Gen3DSR/hunyuan_results/ycb/medium',
                       help='Path to Hunyuan3D results directory')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='Maximum number of scenes to process (default: all)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Analyze directories without generating files')
    
    args = parser.parse_args()
    
    print("Hunyuan3D Results Processor")
    print("="*50)
    print(f"Hunyuan3D path: {args.hunyuan_path}")
    print(f"Max scenes: {args.max_scenes if args.max_scenes else 'All'}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    if not os.path.exists(args.hunyuan_path):
        print(f"Error: Hunyuan3D results path does not exist: {args.hunyuan_path}")
        return
    
    process_hunyuan3d_results(args.hunyuan_path, args.max_scenes, args.dry_run)

if __name__ == "__main__":
    main() 