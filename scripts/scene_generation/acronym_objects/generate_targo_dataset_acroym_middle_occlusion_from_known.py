import os
import argparse
import glob
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import logging
import json
import uuid
from src.vgn.utils.misc import apply_noise
from src.vgn.io import *
from src.vgn.perception import *
from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils.transform import Rotation, Transform
from src.vgn.utils.implicit import get_mesh_pose_dict_from_world

MAX_VIEWPOINT_COUNT = 12
MAX_BIN_COUNT = 1000
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
occ_level_scene_dict = {}
occ_level_dict_count = {
    "0.3-0.4": 0,
    "0.4-0.5": 0,
}

def camera_on_sphere(origin, r, theta, phi):
    """Calculate the camera pose on a sphere centered at origin.
    
    Args:
        origin: The origin of the sphere.
        r: The radius of the sphere.
        theta: The azimuthal angle in the x-y plane from the x-axis.
        phi: The polar angle from the z-axis.
        
    Returns:
        The camera pose as a Transform.
    """
    eye = np.r_[
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ]
    target = origin.translation
    up = np.r_[0.0, 0.0, 1.0]  # this breaks when looking straight down
    return Transform.look_at(eye, target, up)

def render_side_images(sim, n=1, random=False, segmentation=False):
    """渲染场景图像，可以是随机视角或固定视角
    
    Args:
        sim: 模拟环境
        n: 要渲染的图像数量
        random: 是否使用随机视角
        segmentation: 是否生成分割图
        
    Returns:
        depth_imgs: 深度图
        extrinsics: 相机外参
        segs: 分割图（如果segmentation=True）
    """
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    if segmentation:
        segs = np.empty((n, height, width), np.float32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        _, depth_img, seg = sim.camera.render_with_seg(extrinsic, segmentation)

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        segs[i] = seg

    if segmentation:
        return depth_imgs, extrinsics, segs
    else:
        return depth_imgs, extrinsics

def process_and_store_scene_data(sim, scene_id, target_id, noisy_depth_side_c, seg_side_c, extr_side_c, args, occ_level_c):
    """
    Process and store scene data including point clouds and grids.

    Parameters:
    - sim: Simulation instance with necessary methods and camera details.
    - scene_id: Identifier for the scene.
    - target_id: Identifier for the target within the scene.
    - noisy_depth_side_c: Noisy depth image array for the scene.
    - seg_side_c: Segmentation image array for the scene.
    - extr_side_c: Camera extrinsic parameters.
    - args: Namespace containing configuration arguments, including root directory.
    - occ_level_c: Occlusion level of the scene.

    Returns:
    - clutter_id: Constructed identifier for the clutter data.
    """
    # Generate masks from segmentation data
    mask_targ_side_c = seg_side_c == target_id
    mask_scene_side_c = seg_side_c > 0

    # Generate point clouds for target and scene
    pc_targ_side_c = reconstruct_40_pc(sim, noisy_depth_side_c * mask_targ_side_c, extr_side_c)
    if np.asarray(pc_targ_side_c.points, dtype=np.float32).shape[0] == 0:
        return 
    pc_scene_side_c = reconstruct_40_pc(sim, noisy_depth_side_c * mask_scene_side_c, extr_side_c)
    pc_scene_no_targ_side_c = remove_A_from_B(np.asarray(pc_targ_side_c.points, dtype=np.float32),
                                              np.asarray(pc_scene_side_c.points, dtype=np.float32))

    # Depth to point cloud conversions
    pc_scene_depth_side_c = depth_to_point_cloud(noisy_depth_side_c[0], mask_scene_side_c[0],
                                                 sim.camera.intrinsic.K, extr_side_c[0], 2048)
    pc_targ_depth_side_c = depth_to_point_cloud(noisy_depth_side_c[0], mask_targ_side_c[0],
                                                sim.camera.intrinsic.K, extr_side_c[0], 2048)
    pc_scene_no_targ_depth_side_c = remove_A_from_B(pc_targ_depth_side_c, pc_scene_depth_side_c)

    # Generate grids from depth data
    grid_targ_side_c = reconstruct_40_grid(sim, noisy_depth_side_c * mask_targ_side_c, extr_side_c)
    grid_scene_side_c = reconstruct_40_grid(sim, noisy_depth_side_c, extr_side_c)

    # Construct identifier and define output directory
    clutter_id = f"{scene_id}_c_{target_id}"
    test_root = Path(args.output_root) / "scenes_known"

    # Save the processed data
    path = test_root / (clutter_id + ".npz")
    np.savez_compressed(
        path,
        depth_imgs=noisy_depth_side_c,
        extrinsics=extr_side_c,
        mask_targ=mask_targ_side_c.astype(int),
        mask_scene=mask_scene_side_c.astype(int),
        segmentation_map=seg_side_c,
        grid_scene=grid_scene_side_c,
        grid_targ=grid_targ_side_c,
        pc_depth_scene=pc_scene_depth_side_c,
        pc_depth_targ=pc_targ_depth_side_c,
        pc_depth_scene_no_targ=pc_scene_no_targ_depth_side_c,
        pc_scene=np.asarray(pc_scene_side_c.points, dtype=np.float32),
        pc_targ=np.asarray(pc_targ_side_c.points, dtype=np.float32),
        pc_scene_no_targ=pc_scene_no_targ_side_c,
        occ_targ=occ_level_c
    )

    return clutter_id

def depth_to_point_cloud(depth_img, mask_targ, intrinsics, extrinsics, num_points):
    """
    Convert a masked and scaled depth image into a point cloud using camera intrinsics and inverse extrinsics.

    Parameters:
    - depth_img: A 2D numpy array containing depth for each pixel.
    - mask_targ: A 2D boolean numpy array where True indicates the target.
    - intrinsics: The camera intrinsic matrix as a 3x3 numpy array.
    - extrinsics: The camera extrinsic matrix as a 4x4 numpy array. This function assumes the matrix is to be inversed for the transformation.
    - scale: Scale factor to apply to the depth values.

    Returns:
    - A numpy array of shape (N, 3) containing the X, Y, Z coordinates of the points in the world coordinate system.
    """
    # Apply the target mask to the depth image, then apply the scale factor
    depth_img_masked_scaled = depth_img * mask_targ
    
    # Get the dimensions of the depth image
    height, width = depth_img_masked_scaled.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # Flatten the arrays for vectorized operations
    u, v = u.flatten(), v.flatten()
    z = depth_img_masked_scaled.flatten()

    # Convert pixel coordinates (u, v) and depth (z) to camera coordinates
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    
    # Create normal coordinates in the camera frame
    # points_camera_frame = np.array([x, y, z]).T
    points_camera_frame = np.vstack((x, y, z)).T
    points_camera_frame = points_camera_frame[z!=0]
    # Convert the camera coordinates to world coordinate
    # if point_cloud_path is None:
    #     print('point_cloud_path is None')
    points_camera_frame = specify_num_points(points_camera_frame, num_points)

    extrinsic = Transform.from_list(extrinsics).inverse()
    points_transformed = np.array([extrinsic.transform_point(p) for p in points_camera_frame])
    
    return points_transformed

def remove_A_from_B(A, B):
    # Step 1: Use broadcasting to find matching points
    matches = np.all(A[:, np.newaxis] == B, axis=2)
    # Step 2: Identify points in B that are not in A
    unique_to_B = ~np.any(matches, axis=0)
    # Step 3: Filter B to keep only unique points
    B_unique = B[unique_to_B]
    return B_unique

def reconstruct_40_grid(sim, depth_masked, extrinsics):
    """
    Reconstruct a voxel grid from depth images.
    
    Parameters:
    - sim: Simulation instance with camera details
    - depth_masked: Masked depth images
    - extrinsics: Camera extrinsic parameters
    
    Returns:
    - grid_dict: Dictionary containing voxel grid data
    """
    voxel_size = 0.005
    trunc_margin = 0.0025
    
    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=trunc_margin,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )
    
    # Process each depth image and integrate into volume
    for i in range(depth_masked.shape[0]):
        # Skip if the masked depth image has no depth values
        if np.sum(depth_masked[i]) == 0:
            continue
            
        # Convert depth to Open3D image
        depth_img = o3d.geometry.Image(depth_masked[i].astype(np.float32))
        
        # Create intrinsic parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            sim.camera.intrinsic.width,
            sim.camera.intrinsic.height,
            sim.camera.intrinsic.K[0, 0],
            sim.camera.intrinsic.K[1, 1],
            sim.camera.intrinsic.K[0, 2],
            sim.camera.intrinsic.K[1, 2]
        )
        
        # Create extrinsic as 4x4 transformation matrix
        extrinsic = extrinsics[i]
        
        # Integrate depth image into volume
        volume.integrate(depth_img, intrinsic, extrinsic)
    
    # Extract voxel grid from volume
    voxel_grid = volume.extract_voxel_grid()
    
    # Convert to dictionary format
    grid_dict = {
        'origin': voxel_grid.origin,
        'voxel_size': voxel_grid.voxel_size,
        'grid_dimension': voxel_grid.get_dimensions(),
        'voxels': np.asarray(voxel_grid.get_voxels())
    }
    
    return grid_dict

def reconstruct_40_pc(sim, depth_masked, extrinsics):
    """
    Reconstruct a point cloud from depth images.
    
    Parameters:
    - sim: Simulation instance with camera details
    - depth_masked: Masked depth images
    - extrinsics: Camera extrinsic parameters
    
    Returns:
    - point_cloud: Open3D point cloud object
    """
    voxel_size = 0.005
    trunc_margin = 0.0025
    
    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=trunc_margin,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )
    
    # Process each depth image and integrate into volume
    for i in range(depth_masked.shape[0]):
        # Skip if the masked depth image has no depth values
        if np.sum(depth_masked[i]) == 0:
            continue
            
        # Convert depth to Open3D image
        depth_img = o3d.geometry.Image(depth_masked[i].astype(np.float32))
        
        # Create intrinsic parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            sim.camera.intrinsic.width,
            sim.camera.intrinsic.height,
            sim.camera.intrinsic.K[0, 0],
            sim.camera.intrinsic.K[1, 1],
            sim.camera.intrinsic.K[0, 2],
            sim.camera.intrinsic.K[1, 2]
        )
        
        # Create extrinsic as 4x4 transformation matrix
        extrinsic = extrinsics[i]
        
        # Integrate depth image into volume
        volume.integrate(depth_img, intrinsic, extrinsic)
    
    # Extract point cloud from volume
    point_cloud = volume.extract_point_cloud()
    
    return point_cloud

def process_mesh_pose_file(mesh_pose_file, args):
    """从mesh_pose_dict文件创建模拟场景并存储到scenes_known中"""
    
    # 提取文件名中的场景ID
    filename = os.path.basename(mesh_pose_file)
    scene_id = filename.split(".")[0]
    
    # 加载mesh_pose_dict数据
    data = np.load(mesh_pose_file, allow_pickle=True)
    mesh_pose_dict = data['pc'].item()
    target_name = str(data['target_name'])
    
    # 提取目标ID - 从文件名中获取
    # 文件名格式应该是 "scene_id_c_target_id.npz"
    try:
        target_id = int(scene_id.split("_c_")[1])
    except (IndexError, ValueError):
        logger.warning(f"无法从文件名 {filename} 解析目标ID，跳过处理")
        return None
    
    # 创建模拟环境
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui, is_acronym=True)
    
    # 通过mesh_pose_dict创建场景
    logger.info(f"从mesh_pose_dict创建场景: {scene_id}")
    create_scene_from_mesh_pose(sim.world, mesh_pose_dict, sim.object_set)
    
    # 渲染场景获取深度图和分割图
    depth_side_c, extr_side_c, seg_side_c = render_side_images(sim, 1, random=False, segmentation=True)
    
    # 添加噪声到深度图 (参考原始文件，使用原始数据)
    noisy_depth_side_c = depth_side_c
    
    # 获取物体位姿和ID
    target_poses = {}
    target_bodies = {}
    count_cluttered = {}
    
    body_ids = deepcopy(list(sim.world.bodies.keys()))
    body_ids.remove(0)  # 移除平面
    
    # 根据目标ID找到对应的物体
    target_body = None
    for body_id in body_ids:
        body = sim.world.bodies[body_id]
        if body.name == target_name:
            target_id = body_id
            target_body = body
            break
    
    if target_body is None:
        logger.warning(f"在模拟环境中找不到匹配的目标对象: {target_name}")
        sim.disconnect()
        return None
    
    # 计算目标物体的像素数量
    count_cluttered[target_id] = np.count_nonzero(seg_side_c[0] == target_id)
    
    # 移除其他物体，仅保留目标物体和平面
    for body_id in deepcopy(body_ids):
        if body_id != target_id:
            body = sim.world.bodies[body_id]
            sim.world.remove_body(body)
    
    # 渲染仅包含目标物体的场景
    depth_side_s, extr_side_s, seg_side_s = render_side_images(sim, 1, random=False, segmentation=True)
    noisy_depth_side_s = np.array([apply_noise(x, args.add_noise) for x in depth_side_s])
    
    # 计算遮挡级别
    count_single = np.count_nonzero(seg_side_s[0] == target_id)
    if count_single == 0:
        logger.warning(f"目标物体 {target_name} 在单个场景中不可见")
        sim.disconnect()
        return None
        
    occ_level_c = 1 - count_cluttered[target_id] / count_single
    
    # 处理基于遮挡级别的场景
    if 0.3 <= occ_level_c < 0.4:
        bin_key = "0.3-0.4"
    elif 0.4 <= occ_level_c < 0.5:
        bin_key = "0.4-0.5"
    else:
        logger.info(f"场景 {scene_id} 的遮挡级别 {occ_level_c:.2f} 不在目标范围内")
        sim.disconnect()
        return None
    
    # 检查是否超过最大容量
    if occ_level_dict_count[bin_key] >= MAX_BIN_COUNT:
        logger.info(f"遮挡级别 {bin_key} 已达到最大数量 {MAX_BIN_COUNT}")
        sim.disconnect()
        return None
    
    # 重建场景并存储
    # 重新创建完整场景
    for body_id in body_ids:
        if body_id != target_id and body_id != 0:  # 不是目标物体或平面
            sim.world.load_urdf(
                target_bodies[body_id].urdf_path, 
                target_poses[body_id], 
                scale=target_bodies[body_id].scale
            )
    
    # 处理并存储场景数据
    updated_mesh_pose_dict = get_mesh_pose_dict_from_world(sim.world, sim.object_set, exclude_plane=False)
    
    # 确保scenes_known目录存在
    scenes_known_dir = os.path.join(args.output_root, 'scenes_known')
    if not os.path.exists(scenes_known_dir):
        os.makedirs(scenes_known_dir)
    
    # 构建clutter_id
    clutter_id = f"{scene_id}"
    
    # 处理场景数据
    if process_and_store_scene_data(sim, scene_id, target_id, noisy_depth_side_c, seg_side_c, extr_side_c, args, occ_level_c) != None:
        occ_level_dict_count[bin_key] += 1
        occ_level_scene_dict[clutter_id] = occ_level_c
        
        # 保存mesh_pose_dict
        mesh_pose_out_dir = os.path.join(args.output_root, 'scenes_known', 'mesh_pose_dict')
        if not os.path.exists(mesh_pose_out_dir):
            os.makedirs(mesh_pose_out_dir)
        
        mesh_out_path = os.path.join(mesh_pose_out_dir, f"{clutter_id}.npz")
        np.savez_compressed(mesh_out_path, pc=updated_mesh_pose_dict, target_name=target_name)
        
        logger.info(f"处理完成场景 {scene_id}，目标物体 '{target_name}'，遮挡级别: {occ_level_c:.2f}")
    
    # 清理模拟环境
    sim.disconnect()
    return clutter_id

def main(args):
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui, is_acronym=True)
    finger_depth = sim.gripper.finger_depth
    
    # 创建输出目录
    path = f'{args.output_root}/scenes_known'
    if not os.path.exists(path):
        Path(args.output_root).mkdir(parents=True, exist_ok=True)
        os.makedirs(path)
    
    path = f'{args.output_root}/scenes_known/mesh_pose_dict'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 设置记录
    write_setup(
        args.output_root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth
    )
    
    # 获取mesh_pose_dict文件列表
    mesh_pose_path = os.path.join(args.input_root, 'mesh_pose_dict')
    mesh_pose_files = glob.glob(os.path.join(mesh_pose_path, '*.npz'))
    
    if not mesh_pose_files:
        logger.error(f"在 {mesh_pose_path} 中找不到NPZ文件")
        return
    
    logger.info(f"找到 {len(mesh_pose_files)} 个mesh_pose_dict文件需要处理")
    
    # 处理每个文件
    processed_count = 0
    for i, mesh_pose_file in enumerate(mesh_pose_files):
        filename = os.path.basename(mesh_pose_file)
        logger.info(f"处理文件 {i+1}/{len(mesh_pose_files)}: {filename}")
        
        result = process_mesh_pose_file(mesh_pose_file, args)
        if result:
            processed_count += 1
    
    # 断开模拟器连接
    sim.disconnect()
    
    # 保存遮挡级别字典
    occ_level_dict_path = os.path.join(args.output_root, 'scenes_known', 'occ_level_dict.json')
    with open(occ_level_dict_path, "w") as f:
        json.dump(occ_level_scene_dict, f)
    
    logger.info(f"所有处理完成。成功处理 {processed_count}/{len(mesh_pose_files)} 个文件")
    logger.info(f"结果保存到 {args.output_root}/scenes_known")
    logger.info(f"遮挡级别信息保存到 {occ_level_dict_path}")

def farthest_point_sampling(points, num_samples):
    """
    从点云中采样指定数量的点
    
    参数:
      points: 点云数组，形状为 [N, 3]
      num_samples: 采样点数
      
    返回:
      farthest_indices: 采样点的索引
    """
    # 初始化距离数组和结果索引数组
    N, D = points.shape
    farthest_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(N, np.inf)
    
    # 随机选择初始点
    farthest = np.random.randint(0, N)
    
    # 迭代选择最远点
    for i in range(num_samples):
        farthest_indices[i] = farthest
        centroid = points[farthest, :].reshape(1, D)
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    
    return farthest_indices

def specify_num_points(points, target_size):
    """
    确保点云包含指定数量的点
    如果原始点数多于目标点数，使用最远点采样来减少
    如果原始点数少于目标点数，通过复制现有点来增加
    
    参数:
      points: 点云数组，形状为 [N, 3]
      target_size: 目标点数
      
    返回:
      points_specified_num: 调整后的点云
    """
    # 如果点云为空，发出警告
    if points.size == 0:
        print("No points in the scene")
    
    # 如果点数少于目标数量，复制点
    if points.shape[0] < target_size:
        points_specified_num = duplicate_points(points, target_size)
    # 如果点数多于目标数量，使用最远点采样
    elif points.shape[0] > target_size:
        indices = farthest_point_sampling(points, target_size)
        points_specified_num = points[indices]
    else:
        points_specified_num = points
    return points_specified_num

def normalize_pc(pc, scale=1.0):
    """
    对点云进行归一化处理
    
    参数:
      pc: numpy数组，形状为 [N, 3] 的点云数据
      scale: 缩放因子，默认为1.0
      
    返回:
      norm_pc: 归一化后的点云
      centroid: 点云质心
      furthest_distance: 最远点到质心的距离
    """
    # 计算点云质心
    centroid = np.mean(pc, axis=0)
    
    # 将点云中心移到原点
    pc_centered = pc - centroid
    
    # 计算最远点到质心的距离
    furthest_distance = np.max(np.sqrt(np.sum(pc_centered**2, axis=1)))
    
    # 归一化点云
    if furthest_distance > 0:
        norm_pc = pc_centered / furthest_distance * scale
    else:
        norm_pc = pc_centered
        
    return norm_pc, centroid, furthest_distance

def duplicate_points(points, target_size):
    """
    通过重复现有点来达到目标点数
    
    参数:
      points: 原始点云数据
      target_size: 目标点数
      
    返回:
      重复后的点云数据
    """
    repeated_points = points
    while len(repeated_points) < target_size:
        additional_points = points[:min(len(points), target_size - len(repeated_points))]
        repeated_points = np.vstack((repeated_points, additional_points))
    return repeated_points

def check_occ_level_not_full(occ_level_dict):
    """
    检查遮挡级别字典是否已满
    
    参数:
      occ_level_dict: 包含各遮挡级别数量的字典
      
    返回:
      布尔值，表示是否还有未满的遮挡级别
    """
    for occ_level in occ_level_dict:
        if occ_level_dict[occ_level] < MAX_BIN_COUNT:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, 
                       default='/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym-middle-occlusion-1000',
                       help="包含mesh_pose_dict文件夹的根目录")
    parser.add_argument("--output-root", type=str, 
                       default='/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym-middle-occlusion-1000-known',
                       help="将创建scenes_known目录的根目录")
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed",
                       help="场景类型：堆积(pile)或打包(packed)")
    parser.add_argument("--object-set", type=str, default="packed/train",
                       help="物体集合路径")
    parser.add_argument("--save-scene", default=True)
    parser.add_argument("--random", action="store_true", help="添加相机位姿的随机性")
    parser.add_argument("--sim-gui", action="store_true", default=False,
                       help="是否显示模拟器GUI")
    parser.add_argument("--add-noise", type=str, default='norm',
                       help="添加到深度图的噪声类型: norm_0.005 | norm | dex")
    parser.add_argument("--num-proc", type=int, default=2, help="使用的进程数量")
    parser.add_argument("--is-acronym", action="store_true", default=True)
    parser.add_argument("--is-ycb", action="store_true", default=False)
    parser.add_argument("--is-egad", action="store_true", default=False)
    parser.add_argument("--is-g1b", action="store_true", default=False)
    
    args = parser.parse_args()
    
    while check_occ_level_not_full(occ_level_dict_count):
        main(args)
    
    # 保存occlusion level字典
    occ_level_dict_path = f'{args.output_root}/scenes_known/occ_level_dict.json'
    with open(occ_level_dict_path, "w") as f:
        json.dump(occ_level_scene_dict, f)