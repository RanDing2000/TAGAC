#!/usr/bin/env python3
"""
Compare files between mesh_pose_dict and scenes directories.
Find files that exist in mesh_pose_dict but are missing in scenes.
"""

import os
import shutil
import argparse
from tqdm import tqdm


def compare_directories(mesh_pose_dir, scenes_dir, output_file=None, move_missing=False, move_dir=None):
    """
    Compare files in mesh_pose_dict and scenes directories.
    
    Args:
        mesh_pose_dir (str): Path to mesh_pose_dict directory
        scenes_dir (str): Path to scenes directory
        output_file (str): Optional file to save the missing file list
        move_missing (bool): Whether to move missing files to a separate directory
        move_dir (str): Directory to move missing files to
    """
    print(f"Comparing directories:")
    print(f"  Mesh pose dict: {mesh_pose_dir}")
    print(f"  Scenes: {scenes_dir}")
    
    # Get file lists (without .npz extension for comparison)
    mesh_pose_files = set()
    scene_files = set()
    
    # Read mesh_pose_dict files
    if os.path.exists(mesh_pose_dir):
        for f in os.listdir(mesh_pose_dir):
            if f.endswith('.npz'):
                mesh_pose_files.add(f[:-4])  # Remove .npz extension
        print(f"  Found {len(mesh_pose_files)} files in mesh_pose_dict")
    else:
        print(f"  Error: {mesh_pose_dir} does not exist!")
        return
    
    # Read scenes files
    if os.path.exists(scenes_dir):
        for f in os.listdir(scenes_dir):
            if f.endswith('.npz'):
                scene_files.add(f[:-4])  # Remove .npz extension
        print(f"  Found {len(scene_files)} files in scenes")
    else:
        print(f"  Error: {scenes_dir} does not exist!")
        return
    
    # Find differences
    missing_in_scenes = mesh_pose_files - scene_files
    missing_in_mesh_pose = scene_files - mesh_pose_files
    common_files = mesh_pose_files & scene_files
    
    print(f"\nComparison results:")
    print(f"  Common files: {len(common_files)}")
    print(f"  Files in mesh_pose_dict but missing in scenes: {len(missing_in_scenes)}")
    print(f"  Files in scenes but missing in mesh_pose_dict: {len(missing_in_mesh_pose)}")
    
    # Show missing files in scenes
    if missing_in_scenes:
        print(f"\nFiles missing in scenes (first 20):")
        missing_sorted = sorted(missing_in_scenes)
        for i, file in enumerate(missing_sorted[:20]):
            print(f"  {i+1:3d}. {file}")
        
        if len(missing_in_scenes) > 20:
            print(f"  ... and {len(missing_in_scenes) - 20} more files")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Files in mesh_pose_dict but missing in scenes:\n")
                f.write("=" * 50 + "\n")
                for file in missing_sorted:
                    f.write(f"{file}.npz\n")
            print(f"\nComplete list saved to: {output_file}")
        
        # Move missing files if requested
        if move_missing and move_dir:
            print(f"\nMoving missing files to {move_dir}...")
            
            # Create move directory if it doesn't exist
            if not os.path.exists(move_dir):
                os.makedirs(move_dir)
                print(f"Created directory: {move_dir}")
            
            moved_count = 0
            failed_count = 0
            
            for file in tqdm(missing_sorted, desc="Moving files"):
                source_file = os.path.join(mesh_pose_dir, f"{file}.npz")
                dest_file = os.path.join(move_dir, f"{file}.npz")
                
                try:
                    if os.path.exists(source_file):
                        shutil.move(source_file, dest_file)
                        moved_count += 1
                    else:
                        print(f"Warning: Source file not found: {source_file}")
                        failed_count += 1
                except Exception as e:
                    print(f"Error moving {file}.npz: {e}")
                    failed_count += 1
            
            print(f"\nMove completed:")
            print(f"  Successfully moved: {moved_count} files")
            print(f"  Failed: {failed_count} files")
            print(f"  Target directory: {move_dir}")
    
    # Show missing files in mesh_pose_dict
    if missing_in_mesh_pose:
        print(f"\nFiles missing in mesh_pose_dict (first 10):")
        missing_sorted = sorted(missing_in_mesh_pose)
        for i, file in enumerate(missing_sorted[:10]):
            print(f"  {i+1:3d}. {file}")
        
        if len(missing_in_mesh_pose) > 10:
            print(f"  ... and {len(missing_in_mesh_pose) - 10} more files")
    
    return {
        'common': common_files,
        'missing_in_scenes': missing_in_scenes,
        'missing_in_mesh_pose': missing_in_mesh_pose
    }


def main():
    parser = argparse.ArgumentParser(description='Compare mesh_pose_dict and scenes directories')
    parser.add_argument('--mesh_pose_dir', type=str,
                        default='data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000/mesh_pose_dict',
                        help='Path to mesh_pose_dict directory')
    parser.add_argument('--scenes_dir', type=str,
                        default='data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000/scenes',
                        help='Path to scenes directory')
    parser.add_argument('--output_file', type=str,
                        help='Output file to save missing file list')
    parser.add_argument('--move_missing', action='store_true',
                        help='Move missing files to a separate directory')
    parser.add_argument('--move_dir', type=str,
                        default='data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000/mesh_pose_dict_tmp',
                        help='Directory to move missing files to')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    mesh_pose_dir = os.path.abspath(args.mesh_pose_dir)
    scenes_dir = os.path.abspath(args.scenes_dir)
    move_dir = os.path.abspath(args.move_dir) if args.move_dir else None
    
    # Compare directories
    results = compare_directories(mesh_pose_dir, scenes_dir, args.output_file, 
                                args.move_missing, move_dir)


if __name__ == '__main__':
    main() 