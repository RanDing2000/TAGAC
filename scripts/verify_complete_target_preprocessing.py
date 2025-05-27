#!/usr/bin/env python3
"""
Verification script to check if complete target mesh preprocessing was successful.
"""

import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.vgn.io import read_df


def check_scene_complete_target(scene_path):
    """Check if a scene file contains complete target data."""
    try:
        data = np.load(scene_path, allow_pickle=True)
        has_pc = 'complete_target_pc' in data
        has_mesh = 'complete_target_mesh_vertices' in data and 'complete_target_mesh_faces' in data
        
        if has_pc and has_mesh:
            # Check data quality
            pc = data['complete_target_pc']
            vertices = data['complete_target_mesh_vertices']
            faces = data['complete_target_mesh_faces']
            
            pc_valid = pc.shape[0] > 0 and pc.shape[1] == 3
            mesh_valid = vertices.shape[0] > 0 and vertices.shape[1] == 3 and faces.shape[0] > 0
            
            return pc_valid and mesh_valid, pc.shape[0], vertices.shape[0], faces.shape[0]
        else:
            return False, 0, 0, 0
            
    except Exception as e:
        return False, 0, 0, 0


def main(args):
    """Main verification function."""
    dataset_root = Path(args.dataset_root)
    raw_root = Path(args.raw_root)
    
    print("=" * 70)
    print("Complete Target Mesh Preprocessing Verification")
    print("=" * 70)
    print(f"Dataset root: {dataset_root}")
    print(f"Raw root: {raw_root}")
    
    # Read dataset configuration
    try:
        df = read_df(raw_root)
        print(f"Total scenes in dataset: {len(df)}")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return
    
    # Filter for cluttered scenes only
    clutter_scenes = df[df['scene_id'].str.contains('_c_')]['scene_id'].unique()
    print(f"Cluttered scenes to check: {len(clutter_scenes)}")
    
    # Check each scene
    success_count = 0
    total_points = 0
    total_vertices = 0
    total_faces = 0
    
    print("\nChecking scenes...")
    for scene_id in tqdm(clutter_scenes[:args.max_check] if args.max_check > 0 else clutter_scenes):
        scene_path = dataset_root / "scenes" / (scene_id + ".npz")
        
        if not scene_path.exists():
            print(f"Warning: Scene file not found: {scene_path}")
            continue
            
        has_complete, num_points, num_vertices, num_faces = check_scene_complete_target(scene_path)
        
        if has_complete:
            success_count += 1
            total_points += num_points
            total_vertices += num_vertices
            total_faces += num_faces
        elif args.verbose:
            print(f"Missing complete target data: {scene_id}")
    
    # Print results
    print("\n" + "=" * 70)
    print("Verification Results:")
    print("=" * 70)
    print(f"Scenes checked: {len(clutter_scenes[:args.max_check] if args.max_check > 0 else clutter_scenes)}")
    print(f"Scenes with complete target data: {success_count}")
    print(f"Success rate: {success_count/len(clutter_scenes)*100:.1f}%")
    
    if success_count > 0:
        print(f"\nData Quality:")
        print(f"Average points per target: {total_points/success_count:.1f}")
        print(f"Average vertices per mesh: {total_vertices/success_count:.1f}")
        print(f"Average faces per mesh: {total_faces/success_count:.1f}")
    
    # Check a few examples in detail
    if args.detailed and success_count > 0:
        print(f"\nDetailed check of first 3 successful scenes:")
        count = 0
        for scene_id in clutter_scenes:
            if count >= 3:
                break
                
            scene_path = dataset_root / "scenes" / (scene_id + ".npz")
            if not scene_path.exists():
                continue
                
            has_complete, num_points, num_vertices, num_faces = check_scene_complete_target(scene_path)
            if has_complete:
                print(f"\nScene: {scene_id}")
                print(f"  Points: {num_points}")
                print(f"  Vertices: {num_vertices}")
                print(f"  Faces: {num_faces}")
                
                # Load and check data ranges
                data = np.load(scene_path, allow_pickle=True)
                pc = data['complete_target_pc']
                vertices = data['complete_target_mesh_vertices']
                
                print(f"  Point cloud range: [{pc.min():.3f}, {pc.max():.3f}]")
                print(f"  Mesh vertices range: [{vertices.min():.3f}, {vertices.max():.3f}]")
                count += 1
    
    print("=" * 70)
    
    if success_count == len(clutter_scenes):
        print("✓ All scenes have complete target data - ready for training!")
    elif success_count > 0:
        print(f"⚠ {len(clutter_scenes) - success_count} scenes missing complete target data")
        print("Consider re-running preprocessing for missing scenes")
    else:
        print("✗ No complete target data found - preprocessing required")
        print("Run: python scripts/preprocess_complete_target_mesh.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify complete target mesh preprocessing")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to processed dataset root")
    parser.add_argument("--raw_root", type=str, required=True,
                        help="Path to raw dataset root")
    parser.add_argument("--max_check", type=int, default=0,
                        help="Maximum number of scenes to check (0 = all)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed information for sample scenes")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output including missing scenes")
    
    args = parser.parse_args()
    main(args) 