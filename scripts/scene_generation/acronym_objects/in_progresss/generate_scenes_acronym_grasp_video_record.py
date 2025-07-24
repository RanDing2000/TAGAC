## TODO target_id, target_body, mesh_pose_dict should be aligne
import os
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import logging
import json
import uuid
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
from utils_giga import *
from src.vgn.utils.misc import apply_noise
from src.vgn.io import *
from src.vgn.perception import *
from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils.transform import Rotation, Transform
from src.vgn.utils.implicit import get_mesh_pose_dict_from_world, get_mesh_pose_dict_from_world
from src.vgn.grasp import Grasp, Label
import MinkowskiEngine as ME
# import cv2
# import glob

MAX_VIEWPOINT_COUNT = 12
MAX_BIN_COUNT = 1000
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
occ_level_scene_dict = {}
occ_level_dict_count = {
    "0-0.1": 0,
}

def duplicate_points(points, target_size):
    repeated_points = points
    while len(repeated_points) < target_size:
        additional_points = points[:min(len(points), target_size - len(repeated_points))]
        repeated_points = np.vstack((repeated_points, additional_points))
    return repeated_points

def farthest_point_sampling(points, num_samples):
    # Initial farthest point randomly selected
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.full(len(points), np.inf)
    
    for i in range(1, num_samples):
        dist = np.sum((points - farthest_pts[i - 1])**2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest_pts[i] = points[np.argmax(distances)]
    
    return farthest_pts


def specify_num_points(points, target_size):
    # add redundant points if less than target_size
    # if points.shape[0] == 0:
    if points.size == 0:
        print("No points in the scene")
    if points.shape[0] < target_size:
        points_specified_num = duplicate_points(points, target_size)
    # sample farthest points if more than target_size
    elif points.shape[0] > target_size:
        points_specified_num = farthest_point_sampling(points, target_size)
    else:
        points_specified_num = points
    return points_specified_num

def check_occ_level_not_full(occ_level_dict):
    for occ_level in occ_level_dict:
        if occ_level_dict[occ_level] < MAX_BIN_COUNT:
            return True
    return False

def remove_A_from_B(A, B):
    # Step 1: Use broadcasting to find matching points
    matches = np.all(A[:, np.newaxis] == B, axis=2)
    # Step 2: Identify points in B that are not in A
    unique_to_B = ~np.any(matches, axis=0)
    # Step 3: Filter B to keep only unique points
    B_unique = B[unique_to_B]
    return B_unique


def reconstruct_40_grid(sim, depth_imgs, extrinsics):
    tsdf = create_tsdf(sim.size, 40, depth_imgs, sim.camera.intrinsic, extrinsics)
    grid = tsdf.get_grid()
    return grid


def reconstruct_pc(sim, depth_imgs, extrinsics):
    tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
    pc = tsdf.get_cloud()

    # crop surface and borders from point cloud
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    pc = pc.crop(bounding_box)
    if False:
        o3d.visualization.draw_geometries([pc])
    return pc


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6, tgt_id = 0):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width, _ = sim.execute_grasp(candidate, remove=False, tgt_id = tgt_id, force_targ = True)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


def render_images(sim, n,segmentation=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[
                       sim.size / 2, sim.size / 2, 0.0])
    # extrinsics = np.empty((n, 7), np.float32)
    # depth_imgs = np.empty((n, height, width), np.float32)
    # if segmentation:
    #     seg_imgs = np.empty((n, height, width), np.float32)
    extrinsics = np.empty((2*n, 7), np.float32)
    depth_imgs = np.empty((2*n, height, width), np.float32)
    if segmentation:
        seg_imgs = np.empty((2*n, height, width), np.float32)

    for i in range(n):
        for j in range(2):
            r = 2 * sim.size
            theta = (j+1) * np.pi / 6.0
            phi = 2.0 * np.pi * i / n

            extrinsic = camera_on_sphere(origin, r, theta, phi)
            # depth_img = sim.camera.render(extrinsic)[1]
            # if segmentation:
            _, depth_img, seg = sim.camera.render_with_seg(extrinsic, segmentation)


            extrinsics[2*i+j] = extrinsic.to_list()
            depth_imgs[2*i+j] = depth_img
            if segmentation:
                seg_imgs[2*i+j] = seg
    
    if segmentation:
        return depth_imgs, extrinsics, seg_imgs
    else:
        return depth_imgs, extrinsics


def process_test_scene(sim, test_mesh_pose_list, test_scenes, scene_name, curr_mesh_pose_list):
    """
    Process a test scene by loading objects and setting up the simulation environment.
    
    Parameters:
    - sim: Simulation instance
    - test_mesh_pose_list: Directory containing mesh pose lists
    - test_scenes: Directory containing test scenes
    - scene_name: Name of the current scene
    - curr_mesh_pose_list: Current mesh pose list file name
    
    Returns:
    - tgt_id: Target object ID
    - tgt_height: Height of the target object
    - length: Length of the target object
    - width: Width of the target object
    - height: Height of the target object
    - occluder_heights: List of heights of occluder objects
    """
    path_to_npz = os.path.join(test_scenes, curr_mesh_pose_list)
    scene_name = curr_mesh_pose_list[:-4]

    # Prepare simulator
    sim.world.reset()
    sim.world.set_gravity([0.0, 0.0, -9.81])
    sim.draw_workspace()
    sim.save_state()

    # Manually adjust boundaries
    sim.lower = np.array([0.02, 0.02, 0.055])
    sim.upper = np.array([0.28, 0.28, 0.30000000000000004])


    mp_data = np.load(
        os.path.join(test_mesh_pose_list, curr_mesh_pose_list),
        allow_pickle=True
    )["pc"]

    # Place objects
    for obj_id, mesh_info in enumerate(mp_data.item().values()):
        pose = Transform.from_matrix(mesh_info[2])
        # Extract file ID part (without path and extension)
        file_basename = os.path.basename(mesh_info[0])
        if file_basename == 'plane.obj':
            urdf_path = mesh_info[0].replace(".obj", ".urdf")
        else:
            file_id = file_basename.replace("_textured.obj", "").replace(".obj", "")
            
            urdf_base_dir = "/usr/stud/dira/GraspInClutter/targo/data/acronym/urdfs_acronym"
            
            # Method 1: Directly build path (if no category prefix)
            urdf_path = f"{urdf_base_dir}/{file_id}.urdf"
            
            # Method 2: If category prefix exists, use glob to find matching files
            if not os.path.exists(urdf_path):
                import glob
                matching_files = glob.glob(f"{urdf_base_dir}/*_{file_id}.urdf")
                if matching_files:
                    urdf_path = matching_files[0]  # Use the first matching file found
        body = sim.world.load_urdf(
                urdf_path=urdf_path,
                pose=pose,
                scale=mesh_info[1]
            )




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
    pc_scene_depth_side_c_no_specify = depth_to_point_cloud_no_specify(noisy_depth_side_c[0], mask_scene_side_c[0],
                                                 sim.camera.intrinsic.K, extr_side_c[0])
    pc_targ_depth_side_c = depth_to_point_cloud(noisy_depth_side_c[0], mask_targ_side_c[0],
                                                sim.camera.intrinsic.K, extr_side_c[0], 2048)
    pc_targ_depth_side_c_no_specify = depth_to_point_cloud_no_specify(noisy_depth_side_c[0], mask_targ_side_c[0],
                                                sim.camera.intrinsic.K, extr_side_c[0]) 
    pc_scene_no_targ_depth_side_c = remove_A_from_B(pc_targ_depth_side_c, pc_scene_depth_side_c)

    # Generate grids from depth data
    grid_targ_side_c = reconstruct_40_grid(sim, noisy_depth_side_c * mask_targ_side_c, extr_side_c)
    grid_scene_side_c = reconstruct_40_grid(sim, noisy_depth_side_c, extr_side_c)

    # Construct identifier and define output directory
    clutter_id = f"{scene_id}_c_{target_id}"
    # test_root = os.path.join(args.root, 'test_set')
    test_root = args.root
    test_root = Path(test_root)

    # Save the processed data
    write_clutter_sensor_data(
        test_root, clutter_id, noisy_depth_side_c, extr_side_c, mask_targ_side_c.astype(int),
        mask_scene_side_c.astype(int), seg_side_c, grid_scene_side_c, grid_targ_side_c,
        pc_scene_depth_side_c, pc_targ_depth_side_c, pc_scene_no_targ_depth_side_c,
        np.asarray(pc_scene_side_c.points, dtype=np.float32),
        np.asarray(pc_scene_depth_side_c_no_specify, dtype=np.float32),
        np.asarray(pc_targ_side_c.points, dtype=np.float32), 
        np.asarray(pc_targ_depth_side_c_no_specify, dtype=np.float32),
        pc_scene_no_targ_side_c, occ_level_c
    )

    return clutter_id
# Example usage:
# clutter_id = process_and_store_scene_data(sim, scene_id, target_id, noisy_depth_side_c, seg_side_c, extr_side_c, args, occ_level_c)


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

def depth_to_point_cloud_no_specify(depth_img, mask_targ, intrinsics, extrinsics):
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

    extrinsic = Transform.from_list(extrinsics).inverse()
    points_transformed = np.array([extrinsic.transform_point(p) for p in points_camera_frame])
    
    return points_transformed

def render_side_images(sim, n=1, random=False, segmentation=False):
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
        ## export extrinsics matrix as numpy array
        # extrinsics.as_matrix()
        # np.save('extrinsics.npy', extrinsics)
        _, depth_img, seg = sim.camera.render_with_seg(extrinsic, segmentation)

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        segs[i] = seg

    if segmentation:
        return depth_imgs, extrinsics, segs
    else:
        return depth_imgs, extrinsics
        
def main(args):
    # Skip simulation initialization if --no-sim is specified
    if args.no_sim:
        print("Running in no-sim mode, skipping simulation initialization...")
        
        # Create root directory if it doesn't exist
        if not os.path.exists(args.root):
            os.makedirs(args.root, exist_ok=True)
        
        # Create scenes directory if it doesn't exist
        scenes_dir = f'{args.root}/scenes'
        if not os.path.exists(scenes_dir):
            os.makedirs(scenes_dir, exist_ok=True)
        
        # Create test_set directory if it doesn't exist
        test_set_dir = f'{args.root}/test_set'
        if not os.path.exists(test_set_dir):
            os.makedirs(test_set_dir, exist_ok=True)
        
        # Create visualizations directory if it doesn't exist
        vis_dir = "visualization"
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)
        
        print("No-sim mode completed.")
        return
    
    # Normal simulation mode
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui, is_acronym=True)
    finger_depth = sim.gripper.finger_depth
    
    # Write setup for grasp data
    write_setup(
        args.root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth
    )
    
    # Create root directory if it doesn't exist
    if not os.path.exists(args.root):
        os.makedirs(args.root, exist_ok=True)
    
    # Create scenes directory if it doesn't exist
    scenes_dir = f'{args.root}/scenes'
    if not os.path.exists(scenes_dir):
        os.makedirs(scenes_dir, exist_ok=True)
    
    path_scenes_known = f'{args.root}/scenes_known'
    if not os.path.exists(path_scenes_known):
        os.makedirs(path_scenes_known, exist_ok=True)
    
    path_mesh_pose_dict = f'{args.root}/mesh_pose_dict'
    if not os.path.exists(path_mesh_pose_dict):
        os.makedirs(path_mesh_pose_dict, exist_ok=True)

    # Generate scenes first (like in generate_noisy_data_single_clutter.py)
    print("Generating scenes...")
    
    # Generate multiple scenes
    for scene_idx in range(10):  # Generate 10 scenes
        print(f"Generating scene {scene_idx + 1}/10...")
        object_count = np.random.randint(4, 11)  # 11 is excluded
        sim.reset(object_count)
        sim.save_state()
        
        generate_scenes(sim, args)
    
    # Save occ_level_dict for this run
    occ_level_dict_path = f'{args.root}/test_set/occ_level_dict.json'
    with open(occ_level_dict_path, "w") as f:
        json.dump(occ_level_dict_count, f)


def reconstruct_40_pc(sim, depth_imgs, extrinsics):
    tsdf = create_tsdf(sim.size, 40, depth_imgs, sim.camera.intrinsic, extrinsics)
    pc = tsdf.get_cloud()

    # crop surface and borders from point cloud
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    pc = pc.crop(bounding_box)
    if False:
        o3d.visualization.draw_geometries([pc])
    return pc

def generate_scenes(sim, args):
    """
    Generate scenes like in generate_noisy_data_single_clutter.py
    First generate the cluttered scene, then process each target object
    """
    # Generate cluttered scene first
    depth_side_c, extr_side_c, seg_side_c = render_side_images(sim, 1, random=False, segmentation=True)
    noisy_depth_side_c = np.array([apply_noise(x, args.add_noise) for x in depth_side_c])
    depth_c, extr_c, _ = render_images(sim, MAX_VIEWPOINT_COUNT, segmentation=True)
    pc_c = reconstruct_pc(sim, depth_c, extr_c) # point cloud of the clutter scene
    
    if pc_c.is_empty():
        print("Empty point cloud, skipping scene generation")
        return
    
    scene_id = uuid.uuid4().hex
    if args.save_scene:
        mesh_pose_dict = get_mesh_pose_dict_from_world(sim.world, args.object_set, exclude_plane=False)
        write_point_cloud(args.root, scene_id + "_c", mesh_pose_dict, name="mesh_pose_dict")
    
    # get object poses
    target_poses = {}
    target_bodies = {}
    count_cluttered = {}    # count_cluttered is a dictionary that stores the counts of target object
    
    body_ids = deepcopy(list(sim.world.bodies.keys()))
    body_ids.remove(0)  # remove the plane

    for target_id in body_ids:
        assert target_id != 0
        target_body = sim.world.bodies[target_id]   # get the target object
        target_poses[target_id] = target_body.get_pose()
        target_bodies[target_id] = target_body
        count_cluttered[target_id] = np.count_nonzero(seg_side_c[0] == target_id)   # count the number of pixels of the target object in the cluttered scene
    
    # remove all objects first except the plane
    for body_id in body_ids:
        body = sim.world.bodies[body_id]
        sim.world.remove_body(body)
    
    for target_id in body_ids:
        ## make sure a target -> create the single scene -> create the clutter scene
        if count_cluttered[target_id] == 0:  # if the target object is not in the cluttered scene, skip
            continue
        assert target_id != 0 # the plane should not be a target object
        body = target_bodies[target_id]

        #--------------------------------- single scene ---------------------------------##
        target_body = sim.world.load_urdf(body.urdf_path, target_poses[target_id], scale=body.scale)

        if target_id + 1 != target_body.uid:
            print("1")

        ## For body and target_body, the uid changes
        sim.save_state()    # single scene: target object only and plane
        single_id = f"{scene_id}_s_{target_id}"

        ## TODO test mesh pose list and segmentation value is different
        depth_side_s, extr_side_s, seg_side_s = render_side_images(sim, 1, random=False, segmentation=True) # side view is fixed

        noisy_depth_side_s = np.array([apply_noise(x, args.add_noise) for x in depth_side_s])

        count_single = np.count_nonzero(seg_side_s[0] == target_body.uid)
        occ_level_c = 1 - count_cluttered[target_id] / count_single
        if occ_level_c > 0.95:
            curr_body_ids = deepcopy(list(sim.world.bodies.keys()))
            for body_id in curr_body_ids:
                if body_id != 0:
                    body = sim.world.bodies[body_id]
                    sim.world.remove_body(body)
            continue
        mask_targ_side_s = (seg_side_s == target_body.uid)
        depth_s, extr_s, seg_s = render_images(sim, MAX_VIEWPOINT_COUNT, segmentation=True)
        mask_targ_s = (seg_s == target_body.uid)
        mask_scene_side_s = (seg_side_s > 0)
        
        ## TODO, depth points, scene_no_target points, complete target points, filter the size
        pc_targ_side_s = reconstruct_40_pc(sim, noisy_depth_side_s * mask_targ_side_s, extr_side_s) # point cloud of the single scene
        pc_scene_side_s = reconstruct_40_pc(sim, noisy_depth_side_s * mask_scene_side_s, extr_side_s) # point cloud of the single scene
        grid_targ_side_s = reconstruct_40_grid(sim, noisy_depth_side_s * mask_targ_side_s, extr_side_s) # grid of the single scene
        grid_scene_side_s = reconstruct_40_grid(sim, noisy_depth_side_s, extr_side_s) # grid of the single scene
        ## For single scene, there is no scene_no_target points
        pc_scene_depth_side_s = depth_to_point_cloud(noisy_depth_side_s[0],mask_targ_side_s[0], sim.camera.intrinsic.K,extr_side_s[0], 2048)
        pc_targ_depth_side_s = depth_to_point_cloud(noisy_depth_side_s[0],mask_scene_side_s[0], sim.camera.intrinsic.K,extr_side_s[0], 2048)
        pc_s = reconstruct_pc(sim, depth_s, extr_s) # point cloud of the single scene
        # save_point_cloud_as_ply(np.asarray(pc_s.points), 'pc_s.ply')
        if pc_targ_side_s.is_empty():
            curr_body_ids = deepcopy(list(sim.world.bodies.keys()))
            for body_id in curr_body_ids:
                if body_id != 0:
                    body = sim.world.bodies[body_id]
                    sim.world.remove_body(body)
            continue
        quant_points  = ME.utils.sparse_quantize(np.asarray(pc_targ_side_s.points, dtype=np.float32), quantization_size=0.0075)
        if quant_points.shape[0] <= 2:
            curr_body_ids = deepcopy(list(sim.world.bodies.keys()))
            for body_id in curr_body_ids:
                if body_id != 0:
                    body = sim.world.bodies[body_id]
                    sim.world.remove_body(body)
            continue

        complete_target_tsdf = create_tsdf(sim.size, 40, depth_s*mask_targ_s, sim.camera.intrinsic, extr_s) # obtain complete target object tsdf 
        complete_target_pc = complete_target_tsdf.get_cloud()
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        complete_target_pc = complete_target_pc.crop(bounding_box)
        complete_target_pc = np.asarray(complete_target_tsdf.get_cloud().points, dtype=np.float32)
        write_single_scene_data(args.root, single_id,  depth_side_s, extr_side_s, mask_targ_side_s.astype(int),\
                                grid_scene_side_s,grid_targ_side_s, \
                                pc_scene_depth_side_s, pc_targ_depth_side_s, \
                                np.asarray(pc_scene_side_s.points, dtype=np.float32), np.asarray(pc_targ_side_s.points, dtype=np.float32) ,0, complete_target_tsdf.get_grid(), complete_target_pc)
        if args.save_scene:
            # mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set, exclude_plane=False)
            mesh_pose_dict = get_mesh_pose_dict_from_world(sim.world, args.object_set, exclude_plane=False)
            write_point_cloud(args.root, single_id, mesh_pose_dict, name="mesh_pose_dict")
        
        # sample and evaluate grasps for single scene
        grasps = []
        single_outcomes = []
        i = 0
        for _ in range(args.grasps_per_scene):
            point, normal = sample_grasp_point(pc_s, sim.gripper.finger_depth)  # sample a grasp point
            grasp, label = evaluate_grasp_point(sim, point, normal, tgt_id = target_body.uid) # evaluate the grasp point
            grasps.append(grasp)
            single_outcomes.append(label)
            write_grasp(args.root, single_id, grasp, label)  # save grasp for single scene
        
        sim.restore_state() # restore the state of the single scene
        count_single = np.count_nonzero(seg_side_s[0] == target_body.uid)
        
        #--------------------------------- cluttered scene ---------------------------------#
        for other_id in body_ids:
            if other_id != target_id:
                body = target_bodies[other_id]
                other_body = sim.world.load_urdf(body.urdf_path, target_poses[other_id], scale=body.scale)
        sim.save_state()    ## cluttered scene: target object, all the other objects and plane
        
        clutter_id  = f"{scene_id}_c_{target_id}"
        mask_targ_side_c = seg_side_c == target_id
        # mask_targ_side_c = seg_side_c == target_body.uid
        mask_scene_side_c = seg_side_c > 0 
        occ_level_c = 1 - count_cluttered[target_id] / count_single # calculate occlusion level for the target object in cluttered scene
         ## TODO: scene_no_target points
        pc_targ_side_c = reconstruct_40_pc(sim, noisy_depth_side_c * mask_targ_side_c, extr_side_c) # point cloud of the cluttered scene
        # save_point_cloud_as_ply
        # save_point_cloud_as_ply(np.asarray(pc_targ_side_c.points) , 'pc_c.ply')
        pc_scene_side_c = reconstruct_40_pc(sim, noisy_depth_side_c * mask_scene_side_c, extr_side_c) # point cloud of the cluttered scene
        pc_scene_no_targ_side_c = remove_A_from_B(np.asarray(pc_targ_side_c.points,dtype=np.float32), np.asarray(pc_scene_side_c.points,dtype=np.float32))
        pc_scene_depth_side_c = depth_to_point_cloud(noisy_depth_side_c[0],mask_targ_side_c[0], sim.camera.intrinsic.K,extr_side_c[0], 2048)
        pc_targ_depth_side_c = depth_to_point_cloud(noisy_depth_side_c[0],mask_targ_side_c[0], sim.camera.intrinsic.K,extr_side_c[0], 2048)
        pc_scene_no_targ_depth_side_c = remove_A_from_B(pc_targ_depth_side_c, pc_scene_depth_side_c)
        # save_point_cloud_as_ply(pc_targ_depth_side_c , 'pc_depth_c.ply')

        ## TODO: clutter scene should have coorepondence with single scene
        if pc_targ_side_c.is_empty():
            curr_body_ids = deepcopy(list(sim.world.bodies.keys()))
            for body_id in curr_body_ids:
                if body_id != 0:
                    body = sim.world.bodies[body_id]
                    sim.world.remove_body(body)
            continue
        quant_points  = ME.utils.sparse_quantize(np.asarray(pc_targ_side_c.points, dtype=np.float32), quantization_size=0.0075)
        if quant_points.shape[0] <= 2:
            curr_body_ids = deepcopy(list(sim.world.bodies.keys()))
            for body_id in curr_body_ids:
                if body_id != 0:
                    body = sim.world.bodies[body_id]
                    sim.world.remove_body(body)
            continue
        
        grid_targ_side_c = reconstruct_40_grid(sim, noisy_depth_side_c *  mask_targ_side_c, extr_side_c) # grid of the cluttered scene
        grid_scene_side_c = reconstruct_40_grid(sim, noisy_depth_side_c, extr_side_c) # grid of the cluttered scene

        ## TODO,write mask_scene_side_c ?
        write_clutter_sensor_data(args.root, clutter_id,  noisy_depth_side_c, extr_side_c, mask_targ_side_c.astype(int), mask_scene_side_c.astype(int), \
                                seg_side_c,grid_scene_side_c,grid_targ_side_c,\
                                pc_scene_depth_side_c, pc_targ_depth_side_c, pc_scene_no_targ_depth_side_c, \
                                np.asarray(pc_scene_side_c.points,dtype=np.float32), np.asarray(pc_targ_side_c.points,dtype=np.float32), pc_scene_no_targ_side_c, occ_level_c)
        
        
        
        # evaluate grasps from single scene on cluttered scene
        cluttered_outcomes = []
        cluttered_widths = []
        for i, grasp in enumerate(grasps):
            # only execute grasp if success
            if single_outcomes[i] == Label.SUCCESS:
                outcome, width, _ = sim.execute_grasp(grasp, remove=False, tgt_id = target_body.uid, force_targ = True)
            else:
                outcome = Label.FAILURE
                width = sim.gripper.max_opening_width
            # outcome, width = sim.execute_grasp(grasp, remove=False)
            cluttered_outcomes.append(outcome)
            cluttered_widths.append(width)
            write_grasp(args.root, clutter_id, grasp, int(outcome))
            sim.restore_state()
        
        
        ## TODO: remove all the objects after continue
        for body_id in body_ids:
            body = target_bodies[body_id]
            sim.world.remove_body(body)
            
        logger.info(f"scene {scene_id}, target '{target_body.name}' done")

    logger.info(f"scene {scene_id} done")
    return

# def images_to_video(image_dir, output_path, fps=10):
#     images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
#     if not images:
#         print(f'No images found in {image_dir}')
#         return
#     frame = cv2.imread(images[0])
#     height, width, layers = frame.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#     for img_path in images:
#         frame = cv2.imread(img_path)
#         video.write(frame)
#     video.release()
#     print(f'Video saved to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path, default= '/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym_grasp')
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/train")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--save-scene", default=True)
    parser.add_argument("--random", action="store_true", help="Add distribution to camera pose")
    parser.add_argument("--sim-gui", action="store_true", default=False)
    parser.add_argument("--add-noise", type=str, default='norm', help = "norm_0.005 | norm | dex")
    parser.add_argument("--num-proc", type=int, default=2, help="Number of processes to use")
    parser.add_argument("--is-acronym", action="store_true", default=True)
    parser.add_argument("--is-ycb", action="store_true", default=False)
    parser.add_argument("--is-egad", action="store_true", default=False)
    parser.add_argument("--is-g1b", action="store_true", default=False)
    parser.add_argument("--no-sim", action="store_true", help="No simulation mode")
    # parser.add_argument("--record-video", action="store_true", help="Record video in no sim mode")
    # parser.add_argument("--video-output", type=str, default="output.mp4", help="Output video file path")

    args = parser.parse_args()
    
    if args.no_sim:
        # In no-sim mode, just run main once and exit
        main(args)
    else:
        # Normal simulation mode - run in loop until completion
        while check_occ_level_not_full(occ_level_dict_count):
            main(args)

    # Final save of occ_level_dict
    occ_level_dict_path = f'{args.root}/test_set/occ_level_dict.json'
    with open(occ_level_dict_path, "w") as f:
        json.dump(occ_level_dict_count, f)

    # if args.no_sim and args.record_video:
    #     # Create visualizations directory if it doesn't exist
    #     vis_dir = "visualization"
    #     if not os.path.exists(vis_dir):
    #         os.makedirs(vis_dir, exist_ok=True)
    #     images_to_video("visualization", args.video_output)