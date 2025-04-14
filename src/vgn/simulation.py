from pathlib import Path
import time
import numpy as np
import pybullet
import json
import math
import open3d as o3d
import gzip
import matplotlib.pyplot as plt
from typing import Union
from src.utils_targo import points_equal, collect_mesh_pose_dict, find_urdf, adjust_point_cloud_size
from src.vgn.grasp import Label
from src.vgn.perception import TSDFVolume, CameraIntrinsic, create_tsdf, camera_on_sphere
from src.vgn.utils import btsim, workspace_lines
from src.utils_targo import save_point_cloud_as_ply
from src.vgn.utils.transform import Rotation, Transform
import pathlib
import random
import trimesh
import trimesh.transformations as tra
import os

class SceneObject:
    visual_fpath: pathlib.Path
    collision_fpath: pathlib.Path
    pose4x4: np.ndarray = np.eye(4)
    scale: np.ndarray = np.ones(3)
    density: float = 1000.0
    name: str = ""

def load_json(filename: Union[str, pathlib.Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


class YCBPathsLoader:
    ycb_root = Path("/usr/stud/dira/GraspInClutter/targo/data/maniskill_ycb")
    MODEL_JSON = ycb_root / "mani_skill2_ycb/info_pick_v0.json"
    HEAVY_OBJECTS = ["006_mustard_bottle"]


    def __init__(self) -> None:
        if not self.MODEL_JSON.exists():
            raise FileNotFoundError(
                f"json file ({self.MODEL_JSON}) is not found."
                "To download default json:"
                "`python -m mani_skill2.utils.download_asset pick_clutter_ycb`."
            )
        self.model_db: dict[str, dict] = load_json(self.MODEL_JSON)
        self.mesh_names = sorted(list(self.model_db.keys()))
        self.mesh_paths = [
            self.ycb_root / f"mani_skill2_ycb/models/{n}/textured.obj" for n in self.mesh_names   
        ]


    def __len__(self):
        return len(self.model_db)


    def get_random(self) -> pathlib.Path:
        return random.choice(self.mesh_paths)


    def meshpath_to_sceneobj(
        self, meshpath: pathlib.Path, pose: np.ndarray = np.eye(4)
    ) -> SceneObject:
        name = meshpath.parent.stem
        visual_path = str(meshpath)
        collision_path = visual_path.replace("textured.obj", "collision.obj")
        scale = self.model_db[name].get("scales", 1) * np.ones(3)
        density = self.model_db[name].get("density", 1000)
        if name in self.HEAVY_OBJECTS:
            density /= 3
        return SceneObject(
            pathlib.Path(visual_path), pathlib.Path(collision_path), pose, scale, density, name
        )
    

# Simple function to get a single-scene simulator
def sim_select_scene(sim, indices):
    """
    Create a new simulator with only the objects whose indices are specified.
    """
    scene = sim.scene
    object_set = sim.object_set
    sim_selected = ClutterRemovalSim(scene, object_set, False)
    sim.urdf_root = Path("data/urdfs")
    sim.ycb_root = Path("/usr/stud/dira/GraspInClutter/targo/data/urdfs/maniskill_ycb/mani_skill2_ycb/models")
    sim.g1b_root = Path("/usr/stud/dira/GraspInClutter/targo/data/urdfs/g1b/models")
    sim.acroym_root = Path("/usr/stud/dira/GraspInClutter/targo/data/urdfs/acronym/collisions_tabletop")
    # sim.object_urdfs_g1b = [f for f in sim.urdf_root.iterdir() if f.suffix == ".urdf"]

    sim_selected.add_noise = sim.add_noise
    sim_selected.sideview = sim.sideview
    sim_selected.size = sim.size
    intrinsics = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
    sim_selected.camera = sim_selected.world.add_camera(intrinsics, 0.1, 2.0)

    mesh_pose_dict = collect_mesh_pose_dict(sim)
    for idc in indices:
        pose = Transform.from_matrix(mesh_pose_dict[idc][2])
        if idc == 0:
            mesh_path = mesh_pose_dict[idc][0].replace(".obj", ".urdf")
        else:
            mesh_path = find_urdf(mesh_pose_dict[idc][0].replace("_textured.obj", ".urdf"))
        sim_selected.world.load_urdf(mesh_path, pose, mesh_pose_dict[idc][1][0])
    return sim_selected


class ClutterRemovalSim(object):
    """
    A simple class to simulate objects on a table and execute grasps.
    Scenes: 'pile' or 'packed'.
    Models: 'vgn', 'giga_aff', 'giga', 'giga_hr', 'afford_scene_targ_pc' (with fusion_type='targo').
    """

    def __init__(
        self,
        scene,
        object_set,
        size=None,
        gui=False,
        seed=None,
        add_noise=False,
        sideview=False,
        save_dir=None,
        save_freq=8,
        test_root=None,
        is_acronym=False,
        egl_mode=False,
    ):
        # Only allow 'pile' or 'packed' scenes now
        assert scene in ["pile", "packed"]

        self.urdf_root = Path("data/urdfs")
        self.ycb_root = Path("/usr/stud/dira/GraspInClutter/targo/data/maniskill_ycb/mani_skill2_ycb/collisions")
        self.g1b_root = Path("/usr/stud/dira/GraspInClutter/targo/data/g1b/collisions")
        self.egad_root = Path("/usr/stud/dira/GraspInClutter/targo/data/egad/collisions")
        self.acroym_root = Path("/usr/stud/dira/GraspInClutter/targo/data/acronym/urdfs_acronym")
        self.acroym_scales = load_json("data/acronym/acronym_scales.json")
        self.ycb_loader = YCBPathsLoader()
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            "google_pile": 0.7,
            "google_packed": 0.7,
            "acroym": 0.7,
        }.get(object_set, 1.0)

        if is_acronym:
            self.global_scaling = 0.5
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq, egl_mode)
        self.gripper = Gripper(self.world)
        if size:
            self.size = size
        else:
            self.size = 6 * self.gripper.finger_depth

        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

        # For optional occlusion-level bookkeeping
        self.occ_level_dict_path = None
        self.occ_level_dict = {}
        self.save_occ_level_dict = False
        if test_root is not None:
            self.occ_level_dict_path = Path(test_root) / 'test_set' / 'occ_level_dict.json'
            if not self.occ_level_dict_path.exists():
                self.occ_level_dict = {}
                self.save_occ_level_dict = True
            else:
                self.occ_level_dict = json.loads(self.occ_level_dict_path.read_text())
                self.save_occ_level_dict = False

    @property
    def num_objects(self):
        """
        Number of objects (excluding the table).
        """
        return max(0, self.world.p.getNumBodies() - 1)

    def discover_objects(self):
        """
        Collect all the URDF files in the specified folder.
        """
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]
        self.object_urdfs_ycb = [f for f in self.ycb_root.iterdir() if f.name.endswith(".urdf")]
        # self.object_urdfs_g1b = [f for f in self.g1b_root.iterdir() if f.name.endswith(".urdf")]
        self.object_urdfs_g1b = [f for f in self.egad_root.iterdir() if f.name.endswith(".urdf")]
        self.object_urdfs_egad = [f for f in self.egad_root.iterdir() if f.name.endswith(".urdf")]
        self.object_urdfs_acroym = [f for f in self.acroym_root.iterdir() if f.name.endswith(".urdf")]
    def save_state(self):
        """
        Save the simulator state so it can be restored if needed.
        """
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        """
        Restore the simulator to the previously saved state.
        """
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, target_id=None, is_ycb=False, is_g1b=False, is_egad=False, is_acronym=False):
        """
        Reset the world, place the table, and generate a scene with objects.
        """
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        # Generate either pile or packed scene
        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height, target_id=target_id)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height, target_id=target_id, is_ycb=is_ycb, is_g1b=is_g1b, is_egad=is_egad, is_acronym=is_acronym)

    def draw_workspace(self):
        """
        Draw lines marking the valid workspace region in the PyBullet GUI.
        """
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        """
        Load a simple plane for the table and define the valid volume.
        """
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height, target_id=None):
        """
        Generate a 'pile' scene by dropping objects within a box, then removing the box.
        """
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        if target_id is not None:
            random_index = np.random.randint(0, len(urdfs))
            urdfs[random_index] = self.object_urdfs[target_id]
        for urdf_file in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.18, 0.22)
            self.world.load_urdf(urdf_file, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height, target_id=None, is_ycb=False, is_g1b=False, is_egad=False, is_acronym=False):
        """
        Generate a 'packed' scene by placing objects next to each other without overlap.
        
        For Acronym objects, this method follows a similar approach to TableScene.arrange() in the
        acronym_tools library, placing objects in stable poses without collisions.
        """
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            if not is_ycb and not is_g1b and not is_egad and not is_acronym:
                object_urdfs = self.object_urdfs
            elif is_ycb:
                object_urdfs = self.object_urdfs_ycb
            elif is_g1b:
                object_urdfs = self.object_urdfs_g1b
            elif is_egad:
                object_urdfs = self.object_urdfs_egad
            elif is_acronym:
                object_urdfs = self.object_urdfs_acroym
                
            if self.num_objects == 0 and target_id is not None:
                urdf_file = object_urdfs[target_id]
            else:
                urdf_file = self.rng.choice(object_urdfs)

            # For Acronym objects, we try to find stable poses
            if is_acronym:
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                z = table_height + 0.2
                # z = 1.0
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
                pose = Transform(rotation, np.r_[x, y, z])
                
                # Get scale from acronym scales if available
                urdf_id = Path(urdf_file).name
                scale = self.acroym_scales.get(urdf_id, 1.0)
                
                body = self.world.load_urdf(
                    urdf_file, pose, scale=self.global_scaling * scale
                )
                
                lower, upper = self.world.p.getAABB(body.uid)
                z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
                self.world.step()
            elif is_ycb:
                # Original method for non-Acronym objects
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                z = 1.0
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
                pose = Transform(rotation, np.r_[x, y, z])
                
                # scale = self.rng.uniform(0.7, 0.9)
                scale = 1.0
                body = self.world.load_urdf(
                    urdf_file, pose, scale=self.global_scaling * scale
                )
                
                lower, upper = self.world.p.getAABB(body.uid)
                z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
                self.world.step()
            else:
                # Original method for non-Acronym objects
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                z = 1.0
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
                pose = Transform(rotation, np.r_[x, y, z])
                
                scale = self.rng.uniform(0.7, 0.9)
                body = self.world.load_urdf(
                    urdf_file, pose, scale=self.global_scaling * scale
                )
                
                lower, upper = self.world.p.getAABB(body.uid)
                z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
                self.world.step()

            # If it collides or overlaps, remove it
            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

    def acquire_single_tsdf_target_grid(
        self,
        curr_scene_path=None,
        target_id=None,
        resolution=40,
        model="vgn",
        curr_mesh_pose_list=None,
    ):
        """
        Same as acquire_single_tsdf_target_grid_train, but possibly for inference or other usage.
        """
        if model == "giga_hr":
            resolution = 60

        tsdf = TSDFVolume(self.size, resolution)
        tgt_mask_tsdf = TSDFVolume(self.size, resolution)
        plane = np.load("/usr/stud/dira/GraspInClutter/grasping/setup/plane_sampled.npy")
        plane = plane.astype(np.float32)

        half_size = self.size / 2
        origin_yz = np.r_[half_size, half_size]

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, self.size / 3])
            theta, phi = np.pi / 3.0, -np.pi / 2.0
        else:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, 0])
            theta, phi = np.pi / 6.0, 0

        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        _, _, seg_img = self.camera.render_with_seg(extrinsic)

        depth_img = np.load(curr_scene_path)["depth_imgs"]
        tgt_mask = np.load(curr_scene_path)["mask_targ"]
        scene_mask = np.load(curr_scene_path)["mask_scene"]

        seg_img = np.load(curr_scene_path)["segmentation_map"]

        assert np.all(scene_mask == (seg_img > 0))
        assert np.all(tgt_mask == (seg_img == target_id))

        occ_level = 0.0
        if curr_mesh_pose_list is not None:
            if not self.save_occ_level_dict:
                occ_level = self.occ_level_dict[curr_mesh_pose_list]
            else:
                sim_single = sim_select_scene(self, [0, target_id])
                _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
                occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
                self.occ_level_dict[curr_mesh_pose_list] = occ_level

        if occ_level > 0.9:
            print("high occlusion level")

        # print("Occlusion level: ", occ_level)
        # print("Number of objects: ", np.max(seg_img))

        scene_mask = scene_mask.astype(np.uint8)
        tgt_mask = tgt_mask.astype(np.uint8)

        # Keep only these models
        if model in ("vgn", "giga_aff", "giga", "giga_hr"):
            scene_mask = (seg_img >= 0).astype(np.uint8)
        elif model ==  "targo" or model == "targo_full_targ" or model == "targo_hunyun2":
            scene_mask = (seg_img >= 0).astype(np.uint8)

        tic = time.time()
        tsdf.integrate((depth_img * scene_mask)[0], self.camera.intrinsic, extrinsic)
        # tsdf.integrate((depth_img * scene_mask), self.camera.intrinsic, extrinsic)
        tgt_mask_tsdf.integrate((depth_img * tgt_mask)[0], self.camera.intrinsic, extrinsic)
        # tgt_mask_tsdf.integrate((depth_img * tgt_mask), self.camera.intrinsic, extrinsic)
        timing = time.time() - tic

        targ_grid = np.load(curr_scene_path)["grid_targ"]

        # targ_grid = create_tsdf(
        #             self.size,
        #             60,
        #             depth_img * tgt_mask,
        #             self.camera.intrinsic,
        #             np.array(extrinsic.to_list()).reshape(1, 7),
        #         ).get_grid()

        if model in ("vgn", "giga_aff", "giga", "giga_hr"):
            if model != "giga_hr":
                targ_grid = np.load(curr_scene_path)["grid_targ"]
                scene_grid = np.load(curr_scene_path)["grid_scene"]
            else:
                targ_grid = create_tsdf(
                    self.size,
                    60,
                    depth_img * tgt_mask,
                    self.camera.intrinsic,
                    np.array(extrinsic.to_list()).reshape(1, 7),
                ).get_grid()
                scene_grid = create_tsdf(
                    self.size,
                    40,
                    depth_img * scene_mask,
                    self.camera.intrinsic,
                    np.array(extrinsic.to_list()).reshape(1, 7),
                ).get_grid()

            targ_mask = targ_grid > 0
            return tsdf, timing, scene_grid, targ_grid, targ_mask, occ_level

        elif model == "targo" or model == "targo_full_targ" or model == "targo_hunyun2":
            scene_no_targ_pc = np.load(curr_scene_path)["pc_scene_no_targ"]
            targ_depth_pc = np.load(curr_scene_path)["pc_depth_targ"].astype(np.float32)
            save_point_cloud_as_ply(targ_depth_pc, 'targ_depth_pc.ply')
            # scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
            scene_no_targ_pc = scene_no_targ_pc / 0.3 - 0.5
            targ_depth_pc = targ_depth_pc / 0.3 - 0.5
            return tsdf, timing, scene_no_targ_pc, targ_depth_pc,targ_grid, occ_level
        elif model == "FGC-GraspNet" or model == "AnyGrasp":
            scene_pc = np.load(curr_scene_path)['pc_scene_depth_no_specify']
            target_pc = np.load(curr_scene_path)['pc_targ_depth_no_specify']
            # Define a function to adjust point cloud to target size
            # Adjust scene_pc to target size
            # scene_pc = adjust_point_cloud_size(scene_pc)
            # plane = np.load('/usr/stud/dira/GraspInClutter/targo/data/plane.npy')
            # scene_pc = np.concatenate((scene_pc, plane), axis=0)
            # Convert depth image to point cloud for FGC-GraspNet
            # Load depth image and extrinsic parameters
            # depth_imgs = np.load(curr_scene_path)['depth_imgs']
            # scene_mask = np.load(curr_scene_path)['mask_scene']
            # extrinsics = np.load(curr_scene_path)['extrinsics']
            
            # Convert depth to point cloud
            # depth_img = depth_imgs[0]  # (480, 640)
            # extrinsic_tf = Transform.from_list(extrinsics[0])
            # scene_pc = depth_to_point_cloud(depth_img, self.camera.intrinsic, extrinsic_tf)
            # scene_pc shape is (2048, 3)
            # save_point_cloud_as_ply(scene_pc, 'scene_pc_with_plane.ply')  # Visualize using open3d
            ## save
            return tsdf, timing, scene_pc, target_pc, occ_level

        # # Load data
        # depth_img = np.load(curr_scene_path)["depth_imgs"]
        # tgt_mask = np.load(curr_scene_path)["mask_targ"]
        # scene_mask = np.load(curr_scene_path)["mask_scene"]
        
        # # Remove first dimension if needed
        # if len(depth_img.shape) == 3:
        #     depth_img = depth_img[0]  # Convert from (1, 480, 640) to (480, 640)
        # if len(tgt_mask.shape) == 3:
        #     tgt_mask = tgt_mask[0]
        # if len(scene_mask.shape) == 3:
        #     scene_mask = scene_mask[0]
        
        # # Save segmentation image
        # _, _, seg_img = self.camera.render_with_seg(extrinsic)

    def execute_grasp(self, grasp, remove=True, allow_contact=False, tgt_id=0, force_targ=False):
        """
        Execute the given grasp.
        """
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        plan_failure = 0
        visual_failure = 0

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)
        
        # 为视频录制捕获初始位置的几帧 - 减少帧数以提高速度
        if hasattr(self.world, 'recording') and self.world.recording:
            for _ in range(5):  # 从15减少到5
                self.world.capture_frame()

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
            plan_failure = 1
        else:
            # 移动到目标位置
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            
            # 抓取前捕获几帧 - 减少帧数以提高速度
            if hasattr(self.world, 'recording') and self.world.recording:
                for _ in range(5):  # 从15减少到5
                    self.world.capture_frame()
                    
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                # 闭合夹爪
                self.gripper.move(0.0)
                
                # 夹爪闭合时捕获几帧 - 减少帧数以提高速度
                if hasattr(self.world, 'recording') and self.world.recording:
                    for _ in range(5):  # 从15减少到5
                        self.world.capture_frame()
                        
                # 撤回夹爪
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                
                # 结束动作时捕获几帧 - 减少帧数以提高速度
                if hasattr(self.world, 'recording') and self.world.recording:
                    for _ in range(5):  # 从15减少到5
                        self.world.capture_frame()
                        
                if not force_targ:
                    if self.check_success(self.gripper):
                        result = Label.SUCCESS, self.gripper.read()
                        if remove:
                            contacts = self.world.get_contacts(self.gripper.body)
                            self.world.remove_body(contacts[0].bodyB)
                    else:
                        result = Label.FAILURE, self.gripper.max_opening_width
                        visual_failure = 1
                else:
                    res, contacts_targ = self.check_success_target_grasp(self.gripper, tgt_id)
                    if res:
                        result = Label.SUCCESS, self.gripper.read()
                        if remove and contacts_targ:
                            self.world.remove_body(contacts_targ[0].bodyB)
                    else:
                        result = Label.FAILURE, self.gripper.max_opening_width
                        visual_failure = 1

        # 完成后再捕获几帧 - 减少帧数以提高速度
        if hasattr(self.world, 'recording') and self.world.recording:
            for _ in range(5):  # 从15减少到5
                self.world.capture_frame()

        self.world.remove_body(self.gripper.body)
        if remove:
            self.remove_and_wait()

        return result[0], plan_failure, visual_failure

    def remove_and_wait(self):
        """
        Remove any objects that fell outside the workspace and wait for new rest state.
        """
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        """
        Wait until objects are below a velocity threshold or a timeout occurs.
        """
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            for _ in range(60):
                self.world.step()
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        """
        Remove objects that are out of the bounding box.
        """
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        """
        Check if fingers are in contact with any object and not fully closed.
        """
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res

    def check_success_target_grasp(self, gripper, tgt_id=0):
        """
        Check if fingers contact the specific target object.
        """
        contacts = self.world.get_contacts(gripper.body)
        contacts_targ = []
        for contact in contacts:
            if contact.bodyB.uid == tgt_id:
                contacts_targ.append(contact)
        res = len(contacts_targ) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res, contacts_targ

    def check_success_valid(self, gripper, tgt_id=0):
        """
        Optional function for checking valid contact with specific object ID.
        """
        contacts = self.world.get_contacts_valid(gripper.body, tgt_id)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res

    def start_video_recording(self, filename, video_path):
        """
        Start recording a video of the simulation using OpenCV.
        
        Args:
            filename (str): The name of the video file (without extension)
            video_path (str): Path to the directory where the video will be saved
            
        Returns:
            int: Log ID to be used when stopping the recording
        """
        # 使用BtWorld中的OpenCV录制方法
        log_id = self.world.start_video_recording(filename, video_path)
        print(f"开始OpenCV视频录制: {video_path}/{filename}.mp4")
        return log_id
    
    def stop_video_recording(self, log_id):
        """
        Stop an active video recording.
        
        Args:
            log_id (int): The log ID returned by start_video_recording
        """
        # 使用BtWorld中的OpenCV录制方法停止录制
        self.world.stop_video_recording(log_id)
        print("视频录制完成")


class Gripper(object):
    """
    Simple simulated Panda hand with open/close motion.
    """

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        """
        Spawn the gripper in the given pose.
        """
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)

        # Keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)

        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        """
        Update the constraint that holds the gripper's body in place.
        """
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        """
        Directly set the pose of the gripper's Tool Center Point (TCP).
        """
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        """
        Move the gripper in small steps toward a target pose.
        """
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
                # 捕获视频帧
                if hasattr(self.world, 'recording') and self.world.recording:
                    self.world.capture_frame()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        """
        Check contact with any object. If any contact is found, return True.
        """
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        """
        Move fingers to the specified width.
        """
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()
            # 捕获视频帧
            if hasattr(self.world, 'recording') and self.world.recording:
                self.world.capture_frame()

    def read(self):
        """
        Current opening width = sum of the two finger joint positions.
        """
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
