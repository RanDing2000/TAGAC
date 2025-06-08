#!/usr/bin/env python3
"""
为缺少 complete_target_tsdf 的混乱场景生成这些数据
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import trimesh
import sys
import os

# Add project root to path
sys.path.append('/usr/stud/dira/GraspInClutter/targo')

from src.vgn.utils import create_tsdf
from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils import Camera


def load_mesh_from_path(mesh_path):
    """Load mesh from path, handling different formats."""
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    # Load mesh using trimesh
    mesh = trimesh.load(mesh_path)
    
    # Handle Scene objects (convert to single mesh)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Empty scene in mesh file: {mesh_path}")
        # Combine all geometries in the scene
        mesh = mesh.dump(concatenate=True)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load valid mesh from: {mesh_path}")
    
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


def mesh_to_tsdf(mesh, size=0.3, resolution=40):
    """
    Convert mesh to TSDF using the same method as in training.
    
    Args:
        mesh: trimesh.Trimesh object
        size: Size of the workspace
        resolution: TSDF resolution
        
    Returns:
        tsdf: TSDF grid
    """
    try:
        # Create camera and simulation setup
        camera = Camera.default()
        
        # Create TSDF
        tsdf = create_tsdf(
            size=size,
            resolution=resolution,
            depth_imgs=None,  # We don't have depth images, using mesh directly
            intrinsic=camera.intrinsic,
            extrinsics=None
        )
        
        # For mesh-based TSDF, we need to convert mesh to depth images
        # This is a simplified approach - in practice, you might want to render from multiple viewpoints
        
        # Alternative: use mesh voxelization
        voxel_grid = mesh.voxelized(pitch=size/resolution)
        voxel_array = voxel_grid.matrix.astype(np.float32)
        
        # Convert to TSDF format (distance field)
        # This is a simplified conversion - proper TSDF would need signed distance
        tsdf_grid = np.where(voxel_array > 0, 1.0, -1.0)
        
        return tsdf_grid
        
    except Exception as e:
        print(f"Error converting mesh to TSDF: {e}")
        return None


def generate_complete_target_tsdf_for_scene(scene_id, raw_root, output_root):
    """
    Generate complete_target_tsdf for a single scene.
    
    Args:
        scene_id: Scene identifier
        raw_root: Path to raw dataset root
        output_root: Path to output dataset root
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load mesh_pose_dict to get complete target mesh
        mesh_pose_dict_path = raw_root / 'mesh_pose_dict' / (scene_id + '.npz')
        
        if not mesh_pose_dict_path.exists():
            print(f"Warning: mesh_pose_dict not found for {scene_id}")
            return False
        
        mesh_pose_data = np.load(mesh_pose_dict_path, allow_pickle=True)
        mesh_pose_dict = mesh_pose_data['pc'].item()
        
        # Extract target_id from scene_id
        if '_c_' in scene_id:
            target_id = int(scene_id.split('_c_')[-1])
        else:
            print(f"Scene {scene_id} is not a clutter scene, skipping")
            return False
        
        # Generate complete target mesh
        complete_target_mesh, complete_target_pc = get_complete_target_mesh_and_pc(
            mesh_pose_dict, target_id, num_points=2048
        )
        
        if complete_target_mesh is None:
            print(f"Failed to generate complete target mesh for {scene_id}")
            return False
        
        # Convert mesh to TSDF
        complete_target_tsdf = mesh_to_tsdf(complete_target_mesh)
        
        if complete_target_tsdf is None:
            print(f"Failed to generate complete target TSDF for {scene_id}")
            return False
        
        # Load existing scene data
        scene_path = output_root / "scenes" / (scene_id + ".npz")
        if not scene_path.exists():
            print(f"Warning: Scene file not found for {scene_id}")
            return False
        
        existing_data = np.load(scene_path, allow_pickle=True)
        new_data = {key: existing_data[key] for key in existing_data.files}
        
        # Add complete_target_tsdf
        new_data['complete_target_tsdf'] = complete_target_tsdf.astype(np.float32)
        
        # Also ensure complete_target_pc exists (it should from previous preprocessing)
        if 'complete_target_pc' not in new_data:
            new_data['complete_target_pc'] = complete_target_pc
            print(f"Also added missing complete_target_pc for {scene_id}")
        
        # Save updated scene data
        np.savez_compressed(scene_path, **new_data)
        
        print(f"Successfully added complete_target_tsdf for {scene_id}")
        return True
        
    except Exception as e:
        print(f"Error processing scene {scene_id}: {e}")
        return False


def find_scenes_missing_tsdf(dataset_root, max_scenes=0):
    """
    Find cluttered scenes missing complete_target_tsdf.
    
    Args:
        dataset_root: Path to dataset root
        max_scenes: Maximum number of scenes to check (0 = all)
        
    Returns:
        list: List of scene IDs missing complete_target_tsdf
    """
    scenes_dir = Path(dataset_root) / "scenes"
    scene_files = list(scenes_dir.glob("*.npz"))
    
    if max_scenes > 0:
        scene_files = scene_files[:max_scenes]
    
    missing_tsdf_scenes = []
    
    for scene_file in tqdm(scene_files, desc="Scanning for missing TSDF"):
        scene_id = scene_file.stem
        
        # Only check cluttered scenes
        if '_c_' not in scene_id:
            continue
        
        try:
            data = np.load(scene_file, allow_pickle=True)
            
            # Check if complete_target_tsdf is missing
            if 'complete_target_tsdf' not in data:
                missing_tsdf_scenes.append(scene_id)
            elif data['complete_target_tsdf'].size == 0:
                missing_tsdf_scenes.append(scene_id)
                
        except Exception as e:
            print(f"Error reading {scene_id}: {e}")
            missing_tsdf_scenes.append(scene_id)
    
    return missing_tsdf_scenes


def main():
    parser = argparse.ArgumentParser(description="为缺少complete_target_tsdf的混乱场景生成TSDF数据")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="数据集根目录路径")
    parser.add_argument("--raw_root", type=str, default=None,
                        help="原始数据集根目录路径（包含mesh_pose_dict），默认与dataset_root相同")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="最大处理场景数 (0表示处理所有)")
    parser.add_argument("--dry_run", action="store_true",
                        help="只扫描缺失情况，不实际生成数据")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    raw_root = Path(args.raw_root) if args.raw_root else dataset_root
    
    print(f"数据集根目录: {dataset_root}")
    print(f"原始数据集根目录: {raw_root}")
    
    # Find scenes missing complete_target_tsdf
    print("正在扫描缺少 complete_target_tsdf 的混乱场景...")
    missing_scenes = find_scenes_missing_tsdf(dataset_root, args.max_scenes)
    
    print(f"\n找到 {len(missing_scenes)} 个缺少 complete_target_tsdf 的混乱场景")
    
    if not missing_scenes:
        print("所有场景都有 complete_target_tsdf 数据！")
        return
    
    # Show first 10 missing scenes
    print("前10个缺失场景:")
    for scene_id in missing_scenes[:10]:
        print(f"  {scene_id}")
    
    if args.dry_run:
        print("\n干运行模式：只显示缺失情况，不实际生成数据")
        return
    
    # Generate missing TSDF data
    print(f"\n开始为 {len(missing_scenes)} 个场景生成 complete_target_tsdf...")
    
    success_count = 0
    for scene_id in tqdm(missing_scenes, desc="生成TSDF"):
        if generate_complete_target_tsdf_for_scene(scene_id, raw_root, dataset_root):
            success_count += 1
    
    print(f"\n处理完成！")
    print(f"成功处理: {success_count}/{len(missing_scenes)} 个场景")
    if len(missing_scenes) > 0:
        print(f"成功率: {success_count/len(missing_scenes)*100:.1f}%")


if __name__ == "__main__":
    main() 