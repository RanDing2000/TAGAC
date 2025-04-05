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
from src.vgn.utils.misc import apply_noise
from src.vgn.io import *
from src.vgn.perception import *
from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils.transform import Rotation, Transform
from src.vgn.utils.implicit import get_mesh_pose_dict_from_world, get_mesh_pose_dict_from_world

MAX_VIEWPOINT_COUNT = 12
MAX_BIN_COUNT = 1000
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
occ_level_scene_dict = {}
occ_level_dict_count = {
    "0.3-0.4": 0,
    "0.4-0.5": 0,
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
    # test_root = os.path.join(args.root, 'test_set')
    test_root = args.root
    test_root = Path(test_root)

    # Save the processed data
    write_clutter_sensor_data(
        test_root, clutter_id, noisy_depth_side_c, extr_side_c, mask_targ_side_c.astype(int),
        mask_scene_side_c.astype(int), seg_side_c, grid_scene_side_c, grid_targ_side_c,
        pc_scene_depth_side_c, pc_targ_depth_side_c, pc_scene_no_targ_depth_side_c,
        np.asarray(pc_scene_side_c.points, dtype=np.float32),
        np.asarray(pc_targ_side_c.points, dtype=np.float32), pc_scene_no_targ_side_c, occ_level_c
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
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui, is_acronym=True)
    finger_depth = sim.gripper.finger_depth
    
    path = f'{args.root}/scenes'
    if not os.path.exists(path):
        (args.root / "scenes").mkdir(parents=True)
    path = f'{args.root}/mesh_pose_dict'
    if not os.path.exists(path):
        (args.root / "mesh_pose_dict").mkdir(parents=True)
    
    write_setup(
        args.root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth
    )
    if args.save_scene:
        path = f'{args.root}/test_set'
        if not os.path.exists(path):
            os.makedirs(path)
            
    if args.is_acronym:
        object_count = np.random.randint(2, 6)  # 11 is excluded
    else:
        object_count = np.random.randint(4, 11)  # 11 is excluded
    sim.reset(object_count, is_acronym=args.is_acronym, is_ycb=args.is_ycb, is_egad=args.is_egad, is_g1b=args.is_g1b)
    sim.save_state()

    generate_scenes(sim)


def reconstruct_40_pc(sim, depth_imgs, extrinsics):
    tsdf = create_tsdf(sim.size, 40, depth_imgs, sim.camera.intrinsic, extrinsics)
    pc = tsdf.get_cloud()

    # crop surface and borders from point cloud
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    pc = pc.crop(bounding_box)
    if False:
        o3d.visualization.draw_geometries([pc])
    return pc

def generate_scenes(sim):
    depth_side_c, extr_side_c, seg_side_c = render_side_images(sim, 1, random=False, segmentation=True)
    mesh_clutter_pose_dict = get_mesh_pose_dict_from_world(sim.world, sim.object_set, exclude_plane=False)
    ## TODO: add noise to depth image
    # noisy_depth_side_c = np.array([apply_noise(x, args.add_noise) for x in depth_side_c])
    noisy_depth_side_c = depth_side_c
    scene_id = uuid.uuid4().hex
    
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
        if count_cluttered[target_id] == 0:  # if the target object is not in the cluttered scene, skip
            continue
        assert target_id != 0 # the plane should not be a target object
        body = target_bodies[target_id]

        #--------------------------------- single scene ---------------------------------##
        target_body = sim.world.load_urdf(body.urdf_path, target_poses[target_id], scale=body.scale)

        depth_side_s, extr_side_s, seg_side_s = render_side_images(sim, 1, random=False, segmentation=True) # side view is fixed
        noisy_depth_side_s = np.array([apply_noise(x, args.add_noise) for x in depth_side_s])
        
        count_single = np.count_nonzero(seg_side_s[0] == target_body.uid)
        occ_level_c = 1 - count_cluttered[target_id] / count_single

        # Process scenes based on occlusion level
        if 0.3 <= occ_level_c < 0.4:
            bin_key = "0.3-0.4"
        elif 0.4 <= occ_level_c < 0.5:
            bin_key = "0.4-0.5"
        else:
            sim.world.remove_body(target_body)
            continue
            
        current_count = occ_level_dict_count[bin_key]
        if current_count >= MAX_BIN_COUNT:
            sim.world.remove_body(target_body)
            continue
        else:
            if process_and_store_scene_data(sim, scene_id, target_id, noisy_depth_side_c, seg_side_c, extr_side_c, args, occ_level_c) != None:
                occ_level_dict_count[bin_key] += 1
                occ_level_scene_dict[scene_id + '_c_' + str(target_id)] = occ_level_c
                write_test_set_point_cloud(args.root, scene_id + f"_c_{target_id}", mesh_clutter_pose_dict, target_body.name, name="mesh_pose_dict")
        
        sim.world.remove_body(target_body)
        logger.info(f"scene {scene_id}, target '{target_body.name}' done")

    logger.info(f"scene {scene_id} done")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path, default= '/usr/stud/dira/GraspInClutter/targo/output/maniskill-acronym-v2-middle-occlusion-1000')
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

    args = parser.parse_args()
    while check_occ_level_not_full(occ_level_dict_count):
        main(args)

    occ_level_dict_path = f'{args.root}/test_set/occ_level_dict.json'
    with open(occ_level_dict_path, "w") as f:
        json.dump(occ_level_scene_dict, f)