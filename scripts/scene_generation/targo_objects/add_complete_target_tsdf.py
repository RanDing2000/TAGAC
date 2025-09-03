#!/usr/bin/env python3
"""
Script to add complete_target_tsdf to ACRONYM dataset scenes.
Reads complete target mesh data from existing scene files, generates TSDF using mesh_to_tsdf,
and saves the TSDF back to the original scene files.
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

def process_acronym_scenes(dataset_path, max_scenes=None, dry_run=False):
    """
    Process ACRONYM dataset scenes to add complete_target_tsdf.
    
    Args:
        dataset_path: Path to ACRONYM dataset directory
        max_scenes: Maximum number of scenes to process (None for all)
        dry_run: If True, only analyze without modifying files
    """
    scenes_dir = os.path.join(dataset_path, 'scenes')
    
    if not os.path.exists(scenes_dir):
        print(f"Error: Scenes directory not found: {scenes_dir}")
        return
    
    # Get all scene files
    scene_files = [f for f in os.listdir(scenes_dir) if f.endswith('.npz')]
    scene_files.sort()
    
    if max_scenes:
        scene_files = scene_files[:max_scenes]
    
    print(f"Found {len(scene_files)} scene files to process")
    
    # Statistics
    stats = {
        'total_files': len(scene_files),
        'processed': 0,
        'already_has_tsdf': 0,
        'missing_mesh_data': 0,
        'errors': 0,
        'error_files': []
    }
    
    for scene_file in tqdm(scene_files, desc="Processing scenes"):
        scene_path = os.path.join(scenes_dir, scene_file)
        
        try:
            # Load scene data
            with np.load(scene_path, allow_pickle=True) as data:
                scene_data = dict(data)
            
            # Check if complete_target_tsdf already exists
            if 'complete_target_tsdf' in scene_data:
                stats['already_has_tsdf'] += 1
                continue
            
            # Check if required mesh data exists
            if 'complete_target_mesh_vertices' not in scene_data or 'complete_target_mesh_faces' not in scene_data:
                stats['missing_mesh_data'] += 1
                print(f"Warning: Missing mesh data in {scene_file}")
                continue
            
            # Extract mesh data
            vertices = scene_data['complete_target_mesh_vertices']
            faces = scene_data['complete_target_mesh_faces']
            
            # Validate mesh data
            if len(vertices) == 0 or len(faces) == 0:
                stats['missing_mesh_data'] += 1
                print(f"Warning: Empty mesh data in {scene_file}")
                continue
            
            # Create trimesh object
            try:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                # Apply basic mesh fixes using new methods
                mesh.fix_normals()
                
                # Use new methods to clean up mesh
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.update_faces(mesh.unique_faces())
                mesh.remove_unreferenced_vertices()
                
                # Basic validation - check if mesh has vertices and faces after cleanup
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    print(f"Warning: Mesh became empty after cleanup in {scene_file}")
                    stats['errors'] += 1
                    stats['error_files'].append(scene_file)
                    continue
                    
            except Exception as e:
                print(f"Error creating mesh for {scene_file}: {e}")
                stats['errors'] += 1
                stats['error_files'].append(scene_file)
                continue
            
            if dry_run:
                print(f"Would process {scene_file}: vertices={len(vertices)}, faces={len(faces)}")
                stats['processed'] += 1
                continue
            
            # Generate TSDF using mesh_to_tsdf function
            try:
                complete_target_tsdf = mesh_to_tsdf(mesh)
                
                # Validate TSDF
                if complete_target_tsdf is None or complete_target_tsdf.size == 0:
                    print(f"Error: Empty TSDF generated for {scene_file}")
                    stats['errors'] += 1
                    stats['error_files'].append(scene_file)
                    continue
                
                # Add TSDF to scene data
                scene_data['complete_target_tsdf'] = complete_target_tsdf
                
                # Save updated scene data
                np.savez_compressed(scene_path, **scene_data)
                stats['processed'] += 1
                
            except Exception as e:
                print(f"Error generating TSDF for {scene_file}: {e}")
                stats['errors'] += 1
                stats['error_files'].append(scene_file)
                continue
                
        except Exception as e:
            print(f"Error processing {scene_file}: {e}")
            stats['errors'] += 1
            stats['error_files'].append(scene_file)
            continue
    
    # Print statistics
    print("\n" + "="*50)
    print("PROCESSING STATISTICS")
    print("="*50)
    print(f"Total files: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Already had TSDF: {stats['already_has_tsdf']}")
    print(f"Missing mesh data: {stats['missing_mesh_data']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['error_files']:
        print(f"\nError files ({len(stats['error_files'])}):")
        for error_file in stats['error_files'][:10]:  # Show first 10
            print(f"  - {error_file}")
        if len(stats['error_files']) > 10:
            print(f"  ... and {len(stats['error_files']) - 10} more")
    
    print(f"\nSuccess rate: {stats['processed']/(stats['total_files'] - stats['already_has_tsdf'])*100:.1f}%")
    
    # Save error files to txt file if there are any errors
    if stats['error_files']:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_error_files_to_txt(stats['error_files'], script_dir, "acronym")

def save_error_files_to_txt(error_files, script_dir, dataset_name="acronym"):
    """
    Save error filenames to a txt file with timestamp.
    
    Args:
        error_files: List of error filenames
        script_dir: Directory where the script is located
        dataset_name: Name of the dataset for filename
    """
    if not error_files:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_failed_files_{timestamp}.txt"
    filepath = os.path.join(script_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Failed files for {dataset_name.upper()} dataset\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total failed files: {len(error_files)}\n")
            f.write("#" + "="*50 + "\n\n")
            
            for error_file in error_files:
                f.write(f"{error_file}\n")
        
        print(f"Error files list saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Warning: Failed to save error files list: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Add complete_target_tsdf to ACRONYM dataset scenes')
    parser.add_argument('--dataset_path', type=str, 
                       default='data_scenes/targo_dataset',
                       help='Path to ACRONYM dataset directory')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='Maximum number of scenes to process (default: all)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Analyze files without modifying them')
    
    args = parser.parse_args()
    
    print("ACRONYM Complete Target TSDF Generator")
    print("="*50)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Max scenes: {args.max_scenes if args.max_scenes else 'All'}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    process_acronym_scenes(args.dataset_path, args.max_scenes, args.dry_run)

if __name__ == "__main__":
    main() 