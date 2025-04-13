import json
import uuid

import numpy as np
import pandas as pd

from src.vgn.grasp import Grasp
from src.vgn.perception import *
from src.vgn.utils.transform import Rotation, Transform


def write_setup(root, size, intrinsic, max_opening_width, finger_depth):
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict(),
        "max_opening_width": max_opening_width,
        "finger_depth": finger_depth,
    }
    write_json(data, root / "setup.json")

def write_test_set_point_cloud(root, scene_id, point_cloud, target_name,name="test_set"):
    path = root / name /(scene_id + ".npz")
    point_cloud = np.array(point_cloud, dtype=object)
    np.savez_compressed(path, pc=point_cloud, target_name=target_name)

def read_setup(root):
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    max_opening_width = data["max_opening_width"]
    finger_depth = data["finger_depth"]
    return size, intrinsic, max_opening_width, finger_depth


def write_sensor_data(root, depth_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path = root / "scenes" / (scene_id + ".npz")
    assert not path.exists()
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id

def write_single_scene_data(root, scene_id, depth_imgs, extrinsics, mask_targ,  grid_scene = None, grid_targ = None,\
                             pc_depth_scene = None, pc_depth_targ = None,
                             pc_scene = None, pc_targ=None, occ_targ=None, complete_target_tsdf=None, complete_target_pc=None):
    path = root / "scenes" / (scene_id + ".npz")
    assert not path.exists()
    assert '_s_' in scene_id, 'scene_id should have _s_ in it'
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics,
            mask_targ=mask_targ, grid_scene = grid_scene, grid_targ = grid_targ, \
                pc_depth_scene = pc_depth_scene, pc_depth_targ = pc_depth_targ, \
                pc_scene = pc_scene,pc_targ = pc_targ,occ_targ=occ_targ, complete_target_tsdf = complete_target_tsdf, complete_target_pc = complete_target_pc)

def write_double_scene_data(root, scene_id, depth_imgs, extrinsics, mask_targ, grid_scene = None, grid_targ = None, pc_scene = None, pc_targ=None, occ_targ=None):
    path = root / "scenes" / (scene_id + ".npz")
    assert not path.exists()
    assert '_d_' in scene_id, 'scene_id should have _d_ in it'
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, mask_targ=mask_targ,\
                            grid_scene = grid_scene, grid_targ = grid_targ, pc_scene = pc_scene, pc_targ = pc_targ,occ_targ=occ_targ)
def write_clutter_sensor_data(root, scene_id, depth_imgs, extrinsics, mask_targ, mask_scene,  \
                               segmentation_map = None, grid_scene = None, grid_targ = None,\
                               pc_depth_scene = None, pc_depth_targ = None, pc_depth_scene_no_targ = None,
                               pc_scene = None, pc_scene_depth_no_specify = None,
                               pc_targ = None, pc_targ_depth_no_specify = None,
                               pc_scene_no_targ = None, occ_targ=None):
    path = root / 'scenes' / (scene_id + ".npz")
    assert not path.exists()
    assert '_c_' in scene_id, 'scene_id should have _c_ in it'
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, mask_targ=mask_targ, mask_scene=mask_scene, \
                            segmentation_map = segmentation_map, grid_scene = grid_scene, grid_targ = grid_targ,\
                            pc_depth_scene = pc_depth_scene, pc_depth_targ = pc_depth_targ, pc_depth_scene_no_targ = pc_depth_scene_no_targ,\
                            pc_scene = pc_scene, pc_scene_depth_no_specify = pc_scene_depth_no_specify,\
                            pc_targ = pc_targ, pc_targ_depth_no_specify = pc_targ_depth_no_specify,\
                            pc_scene_no_targ = pc_scene_no_targ, occ_targ=occ_targ)

def write_full_sensor_data(root, depth_imgs, extrinsics, scene_id=None):
    if scene_id is None:
        scene_id = uuid.uuid4().hex
    path = root / "full_scenes" / (scene_id + ".npz")
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def read_sensor_data(root, scene_id):
    data = np.load(root / "scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]

def read_full_sensor_data(root, scene_id):
    data = np.load(root / "full_scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]


def write_grasp(root, scene_id, grasp, label):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)


def read_grasp(df, i):
    scene_id = df.loc[i, "scene_id"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, grasp, label


def read_df(root):
    return pd.read_csv(root / "grasps.csv")


def write_df(df, root):
    df.to_csv(root / "grasps.csv", index=False)


def write_voxel_grid(root, scene_id, voxel_grid):
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, grid=voxel_grid)


def write_point_cloud(root, scene_id, point_cloud, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    np.savez_compressed(path, pc=point_cloud)

def read_voxel_grid(root, scene_id):
    path = root / "scenes" / (scene_id + ".npz")
    return np.load(path)["grid"]

def read_point_cloud(root, scene_id, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    return np.load(path)["pc"]

def read_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")
