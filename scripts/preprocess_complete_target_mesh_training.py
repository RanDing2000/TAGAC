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

from src.vgn.io import read_df, read_setup
from src.vgn.utils.transform import Transform


def load_mesh_from_path(mesh_path):
    """Load mesh from file path, handling both .urdf and mesh files."""
    if os.path.splitext(mesh_path)[1] == '.urdf':
        obj = URDF.load(mesh_path)
        assert len(obj.links) == 1, "Assumption that URDF has exactly one link might not hold true."
        assert len(obj.links[0].visuals) == 1, "Assumption that each link has exactly one visual might not hold true."
        assert len(obj.links[0].visuals[0].geometry.meshes) == 1, "Assumption that each visual has exactly one mesh might not hold true."
        mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
    else:
        mesh = trimesh.load(mesh_path)
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
        # Read mesh_pose_dict for this scene
        if '_c_' in scene_id:
            clutter_scene_id = scene_id.split('_c_')[0]
            mesh_pose_dict_path = raw_root / 'mesh_pose_dict' / (clutter_scene_id + '_c.npz')
        elif '_s_' in scene_id:
            single_scene_id = scene_id
            mesh_pose_dict_path = raw_root / 'mesh_pose_dict' / (single_scene_id + '.npz')
        else:
            print(f"Warning: Cannot extract target_id from scene_id {scene_id}")
            return False

        if not mesh_pose_dict_path.exists():
            assert False, f"mesh_pose_dict not found for {scene_id}"

        mesh_pose_data = np.load(mesh_pose_dict_path, allow_pickle=True)
        mesh_pose_dict = mesh_pose_data['pc'].item()

        # Extract target_id
        if '_c_' in scene_id:
            target_id = int(scene_id.split('_c_')[-1])
        elif '_s_' in scene_id:
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

        # Load target point cloud
        scene_data = np.load(output_root / "scenes" / (scene_id + ".npz"), allow_pickle=True)
        target_data = scene_data['pc_targ']

        # Assert target_data ⊂ complete_target_pc (in nearest neighbor sense)
        # from scipy.spatial import cKDTree
        # tree = cKDTree(complete_target_pc)
        # distances, _ = tree.query(target_data, k=1)
        # assert np.all(distances < 1e-3), (
        #     f"target_pc is not a subset of complete_target_pc (max dist: {distances.max()})"
        # )

        # Optional: visualize and save
        import open3d as o3d
        complete_target_pc_o3d = o3d.geometry.PointCloud()
        complete_target_pc_o3d.points = o3d.utility.Vector3dVector(complete_target_pc)
        target_pc_o3d = o3d.geometry.PointCloud()
        target_pc_o3d.points = o3d.utility.Vector3dVector(target_data)
        o3d.io.write_point_cloud("complete_target_pc.ply", complete_target_pc_o3d)
        o3d.io.write_point_cloud("target_pc.ply", target_pc_o3d)

        # Save updated scene data with complete mesh
        scene_path = output_root / "scenes" / (scene_id + ".npz")
        if not scene_path.exists():
            print(f"Warning: Scene file not found for {scene_id}")
            return False

        existing_data = np.load(scene_path, allow_pickle=True)
        new_data = {key: existing_data[key] for key in existing_data.files}
        new_data['complete_target_pc'] = complete_target_pc
        new_data['complete_target_mesh_vertices'] = complete_target_mesh.vertices.astype(np.float32)
        new_data['complete_target_mesh_faces'] = complete_target_mesh.faces.astype(np.int32)
        np.savez_compressed(scene_path, **new_data)

        return True

    except Exception as e:
        print(f"Error processing scene {scene_id}: {e}")
        return False



def main(args):
    """Main preprocessing function."""
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    
    print(f"Raw dataset root: {raw_root}")
    print(f"Output dataset root: {output_root}")
    
    # Read dataset configuration
    df = read_df(raw_root)
    
    # Filter for both clutter and single scenes
    # Clutter scenes: contain '_c_' (format: scene_c_target_id)
    # Single scenes: contain '_s_' (format: scene_s_object_id_target_id)
    clutter_scenes = df[df['scene_id'].str.contains('_c_')]['scene_id'].unique()
    single_scenes = df[df['scene_id'].str.contains('_s_')]['scene_id'].unique()
    
    # Combine both types of scenes
    all_scenes = np.concatenate([clutter_scenes, single_scenes])
    # all_scenes = single_scenes
    
    print(f"Found {len(clutter_scenes)} clutter scenes")
    print(f"Found {len(single_scenes)} single scenes")
    print(f"Total {len(all_scenes)} scenes to process")
    
    # Process each scene
    success_count = 0
    total_count = len(all_scenes)
    
    for scene_id in tqdm(all_scenes, desc="Processing scenes"):
        if process_scene(scene_id, raw_root, output_root):
            success_count += 1
        
        # Optional: Process only a subset for testing
        if args.max_scenes > 0 and success_count >= args.max_scenes:
            break
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{total_count} scenes")
    print(f"Success rate: {success_count/total_count*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess complete target meshes for training dataset")
    parser.add_argument("--raw_root", type=str, default='/home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset',
                        help="Path to raw dataset root containing mesh_pose_dict")
    parser.add_argument("--output_root", type=str, default='/home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset',
                        help="Path to output dataset root where scenes are stored")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="Maximum number of scenes to process (0 = all)")
    
    args = parser.parse_args()
    main(args) 