#!/usr/bin/env python3
"""
Offline preprocessing script to generate complete target meshes from mesh_pose_dict
and save them to the training dataset.

This script reads mesh_pose_dict files and generates complete target meshes and point clouds,
then saves them to the scenes directory for use during training.

Supports both clutter scenes (format: scene_c_target_id) and single scenes (format: scene_s_object_id_target_id).
"""

import argparse
import os
import numpy as np
import trimesh
from pathlib import Path
from urdfpy import URDF
import pandas as pd
from tqdm import tqdm

# Remove dependency on read_df since grasps.csv may not exist
# from src.vgn.io import read_df, read_setup


def load_mesh_from_path(mesh_path):
    """Load mesh from file path, handling both .urdf and mesh files."""
    if os.path.splitext(mesh_path)[1] == '.urdf':
        obj = URDF.load(mesh_path)
        assert len(obj.links) == 1, "Assumption that URDF has exactly one link might not hold true."
        assert len(obj.links[0].visuals) == 1, "Assumption that each link has exactly one visual might not hold true."
        assert len(obj.links[0].visuals[0].geometry.meshes) == 1, "Assumption that each visual has exactly one mesh might not hold true."
        mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
    else:
        loaded_obj = trimesh.load(mesh_path)
        
        # Handle case where trimesh.load returns a Scene object instead of Mesh
        if isinstance(loaded_obj, trimesh.Scene):
            # If it's a scene, try to get the first mesh from the scene
            if len(loaded_obj.geometry) == 0:
                raise ValueError(f"Scene has no geometry: {mesh_path}")
            elif len(loaded_obj.geometry) == 1:
                # Single mesh in scene
                mesh = list(loaded_obj.geometry.values())[0]
            else:
                # Multiple meshes in scene, combine them
                mesh = loaded_obj.dump(concatenate=True)
        else:
            # Already a Mesh object
            mesh = loaded_obj
    
    return mesh


def get_complete_target_mesh_and_pc(mesh_pose_dict, target_id, num_points=2048):
    """
    Generate complete target mesh and point cloud from mesh_pose_dict.
    
    Args:
        mesh_pose_dict: Dictionary containing mesh paths, scales, and poses
        target_id: ID of the target object
        num_points: Number of points to sample from mesh surface
        
    Returns:
        target_mesh: Complete target mesh as trimesh object
        target_pc: Target point cloud sampled from complete mesh
    """
    if target_id not in mesh_pose_dict:
        print(f"Warning: target_id {target_id} not found in mesh_pose_dict")
        return None, None
        
    mesh_path, scale, pose = mesh_pose_dict[target_id]
    
    try:
        # Load the mesh
        target_mesh = load_mesh_from_path(mesh_path)
        
        # Apply scale and pose transformations
        target_mesh.apply_scale(scale)
        target_mesh.apply_transform(pose)
        
        # Sample points from the complete mesh
        target_pc, _ = trimesh.sample.sample_surface(target_mesh, num_points)
        target_pc = target_pc.astype(np.float32)
        
        return target_mesh, target_pc
        
    except Exception as e:
        print(f"Error loading mesh from {mesh_path}: {e}")
        return None, None


def process_scene(scene_id, raw_root, output_root):
    """
    Process a single scene to generate complete target mesh and point cloud.

    Args:
        scene_id: Scene identifier 
                 - Clutter scene format: "scene_c_target_id" 
                 - Single scene format: "scene_s_object_id_target_id"
        raw_root: Path to raw dataset root
        output_root: Path to output dataset root

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # For both clutter and single scenes, the mesh_pose_dict file has the same name as the scene file
        mesh_pose_dict_path = raw_root / 'mesh_pose_dict' / (scene_id + '.npz')

        if not mesh_pose_dict_path.exists():
            print(f"Warning: mesh_pose_dict not found for {scene_id}: {mesh_pose_dict_path}")
            return False

        mesh_pose_data = np.load(mesh_pose_dict_path, allow_pickle=True)
        mesh_pose_dict = mesh_pose_data['pc'].item()

        # Extract target_id from scene_id
        if '_c_' in scene_id:
            # Clutter scene: format is "scene_id_c_target_id"
            target_id = int(scene_id.split('_c_')[-1])
        elif '_s_' in scene_id:
            # Single scene: format is "scene_id_s_object_id_target_id" 
            # For single scenes, we use the highest numbered object as target
            target_id = int(max(mesh_pose_dict.keys()))
        else:
            print(f"Warning: Cannot extract target_id from scene_id {scene_id}")
            return False

        # Generate complete target mesh and point cloud
        complete_target_mesh, complete_target_pc = get_complete_target_mesh_and_pc(
            mesh_pose_dict, target_id, num_points=2048
        )

        if complete_target_mesh is None or complete_target_pc is None:
            print(f"Failed to generate complete target for {scene_id}")
            return False

        # Save updated scene data with complete mesh
        scene_path = output_root / "scenes" / (scene_id + ".npz")
        if not scene_path.exists():
            print(f"Warning: Scene file not found for {scene_id}: {scene_path}")
            return False

        # Load existing scene data
        existing_data = np.load(scene_path, allow_pickle=True)
        new_data = {key: existing_data[key] for key in existing_data.files}

        scene_data = np.load(output_root / "scenes" / (scene_id + ".npz"), allow_pickle=True)
        target_data = scene_data['pc_targ']

        # Optional: visualize and save
        import open3d as o3d
        complete_target_pc_o3d = o3d.geometry.PointCloud()
        complete_target_pc_o3d.points = o3d.utility.Vector3dVector(complete_target_pc)
        target_pc_o3d = o3d.geometry.PointCloud()
        target_pc_o3d.points = o3d.utility.Vector3dVector(target_data)
        o3d.io.write_point_cloud("complete_target_pc_new.ply", complete_target_pc_o3d)
        o3d.io.write_point_cloud("target_pc_new.ply", target_pc_o3d)

        print(f"complete_target_pc.shape: {complete_target_pc.shape}")
        print(f"target_data.shape: {target_data.shape}")
        
        # Add complete target data
        new_data['complete_target_pc'] = complete_target_pc
        new_data['complete_target_mesh_vertices'] = complete_target_mesh.vertices.astype(np.float32)
        new_data['complete_target_mesh_faces'] = complete_target_mesh.faces.astype(np.int32)
        
        # Save updated scene data
        np.savez_compressed(scene_path, **new_data)

        return True

    except Exception as e:
        print(f"Error processing scene {scene_id}: {e}")
        return False


def get_scene_list_from_directory(scenes_dir):
    """
    Get list of scene IDs from the scenes directory.
    
    Args:
        scenes_dir: Path to scenes directory
        
    Returns:
        list: List of scene IDs (without .npz extension)
    """
    scenes_dir = Path(scenes_dir)
    if not scenes_dir.exists():
        print(f"Warning: Scenes directory does not exist: {scenes_dir}")
        return []
    
    scene_files = list(scenes_dir.glob("*.npz"))
    scene_ids = [f.stem for f in scene_files]
    
    return scene_ids


def main(args):
    """Main preprocessing function."""
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    
    print(f"Raw dataset root: {raw_root}")
    print(f"Output dataset root: {output_root}")
    
    # Get scene list directly from scenes directory instead of reading grasps.csv
    scenes_dir = output_root / "scenes"
    all_scene_ids = get_scene_list_from_directory(scenes_dir)
    
    if not all_scene_ids:
        print("No scene files found in scenes directory!")
        return
    
    # Filter for both clutter and single scenes
    clutter_scenes = [s for s in all_scene_ids if '_c_' in s]
    single_scenes = [s for s in all_scene_ids if '_s_' in s]
    other_scenes = [s for s in all_scene_ids if '_c_' not in s and '_s_' not in s]
    
    print(f"Found {len(clutter_scenes)} clutter scenes")
    print(f"Found {len(single_scenes)} single scenes") 
    print(f"Found {len(other_scenes)} other scenes")
    print(f"Total {len(all_scene_ids)} scenes to process")
    
    # Process scenes that match our expected formats
    scenes_to_process = clutter_scenes + single_scenes
    if not scenes_to_process:
        print("No clutter or single scenes found! Processing all scenes...")
        scenes_to_process = all_scene_ids
    
    # Process each scene
    success_count = 0
    total_count = len(scenes_to_process)
    
    for scene_id in tqdm(scenes_to_process, desc="Processing scenes"):
        if process_scene(scene_id, raw_root, output_root):
            success_count += 1
        
        # Optional: Process only a subset for testing
        if args.max_scenes > 0 and success_count >= args.max_scenes:
            break
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{total_count} scenes")
    if total_count > 0:
        print(f"Success rate: {success_count/total_count*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess complete target meshes for training dataset")
    parser.add_argument("--raw_root", type=str, default='/storage/user/dira/nips_data_version6/combined/targo_dataset',
                        help="Path to raw dataset root containing mesh_pose_dict")
    parser.add_argument("--output_root", type=str, default='/storage/user/dira/nips_data_version6/combined/targo_dataset',
                        help="Path to output dataset root where scenes are stored")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="Maximum number of scenes to process (0 = all)")
    
    args = parser.parse_args()
    main(args) 