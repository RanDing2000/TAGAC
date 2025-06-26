import time
import numpy as np
import trimesh
from scipy import ndimage
import torch
import os

from src.vgn.grasp import *
from src.vgn.utils.transform import Transform, Rotation
from src.vgn.networks import load_network
from src.vgn.utils import visual
from src.utils_giga import *
from src.utils_targo import tsdf_to_mesh, filter_and_pad_point_clouds, save_point_cloud_as_ply

LOW_TH = 0.0

def predict_ptv3_scene(inputs, pos, net, device, visual_dict=None, state=None, target_mesh_gt=None):
    """
    Predict function specialized for ptv3_scene model type.
    
    Args:
        inputs: tuple of (scene_point_cloud, None) where scene_point_cloud includes labels
        pos: position tensor for queries
        net: ptv3_scene network model
        device: torch device
        visual_dict: dictionary for visualization data
        state: state object containing preprocessed complete target data
        target_mesh_gt: ground truth target mesh (fallback option)
        
    Returns:
        qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou
    """
    # scene_no_targ_pc = torch.from_numpy(inputs[0]).unsqueeze(0).to(device)
    full_scene_with_labels = inputs[0]

    with torch.no_grad():
        # For ptv3_scene: directly use preprocessed complete target data
        start_pc = time.time()
        # Read complete target TSDF if available
        # if hasattr(state, 'complete_target_tsdf') and state.complete_target_tsdf is not None:
        #     # Use preprocessed complete target TSDF
        completed_targ_grid = state.complete_targ_tsdf
    
        # Perfect reconstruction means CD and IoU are ideal (since using ground truth/preprocessed data)
        cd = state.cd
        iou = state.iou
        
        # completed_targ_pc  = state.complete_targ_pc
        
        
        # save_point_cloud_as_ply(targ_completed_scene_pc[0].cpu().numpy(), 'targ_completed_scene_pc.ply')
        
        convert_time = time.time() - start_pc
        print(f"convert: {convert_time:.3f}s")
        
        # For ptv3_scene, only pass scene point cloud
        # model_inputs = (full_scene_with_labels, None)
        full_scene_with_labels_tensor = torch.from_numpy(full_scene_with_labels).unsqueeze(0).to(device)
        model_inputs = full_scene_with_labels_tensor
        save_point_cloud_as_ply(full_scene_with_labels_tensor[:, :,:3][0].cpu().numpy(), 'full_scene_with_labels_tensor.ply')
    
    time_grasp_start = time.time()
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(model_inputs, pos)
        
        net_params_count = sum(p.numel() for p in net.parameters())
        print(f"Number of parameters in self.net: {net_params_count:,}")
    
    time_grasp = time.time() - time_grasp_start
    print(f"Grasp prediction time: {time_grasp:.3f}s")
    
    # Move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    
    return qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5
):
    """Process prediction volumes for ptv3_scene model."""
    # Check if tsdf_vol is a tuple
    if isinstance(tsdf_vol, tuple):
        if len(tsdf_vol) == 2:
            tsdf_vol = tsdf_vol[0]
    tsdf_vol = tsdf_vol.squeeze()
    
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # Mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # Reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    """Select grasps from prediction volumes."""
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # Threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # Non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # Construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]
        
    return sorted_grasps, sorted_scores


def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    """Select grasp at specific index."""
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score


def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    """Avoid grasp out of bound."""
    # avoid grasp out of bound
    x, y, z = np.mgrid[:qual_vol.shape[0], :qual_vol.shape[1], :qual_vol.shape[2]]
    xyz = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T 
    xyz = xyz * voxel_size  + limit
    # xyz = xyz * voxel_size  + [0.02, 0.02, 0.055]
    # filter out all grasps outside of the boundary
    # mask = np.logical_and(0.02 <= xyz[:, 0], xyz[:, 0] <= 0.28)
    # mask = np.logical_and(mask, np.logical_and(0.02 <= xyz[:, 1], xyz[:, 1] <= 0.28))
    # mask = np.logical_and(mask, np.logical_and(0.055 <= xyz[:, 2], xyz[:, 2] <= 0.30))
    mask = np.logical_and(limit[0] <= xyz[:, 0], xyz[:, 0] <= 0.28)
    mask = np.logical_and(mask, np.logical_and(limit[1] <= xyz[:, 1], xyz[:, 1] <= 0.28))
    mask = np.logical_and(mask, np.logical_and(limit[2] <= xyz[:, 2], xyz[:, 2] <= 0.30))
    mask = mask.reshape(qual_vol.shape)
    qual_vol[~mask] = 0.0
    return qual_vol


class PTV3SceneImplicit(object):
    """
    Implicit grasp detection class specialized for ptv3_scene model type.
    This class directly uses preprocessed complete target point clouds and TSDF data.
    """
    
    def __init__(self, model_path, model_type, best=False, force_detection=False, 
                 qual_th=0.9, out_th=0.5, visualize=False, resolution=40, 
                 cd_iou_measure=False, **kwargs):
        """
        Initialize PTV3SceneImplicit detector.
        
        Args:
            model_path: path to the trained model
            model_type: must be 'ptv3_scene'
            best: whether to return only the best grasp
            force_detection: whether to force detection even with low quality
            qual_th: quality threshold for grasp selection
            out_th: output threshold for processing
            visualize: whether to enable visualization
            resolution: voxel grid resolution
            cd_iou_measure: whether to measure Chamfer Distance and IoU
        """
        assert model_type == 'ptv3_scene', f"PTV3SceneImplicit only supports ptv3_scene model type, got {model_type}"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type) 
        self.net = self.net.eval()
        
        net_params_count = sum(p.numel() for p in self.net.parameters())
        print(f"Number of parameters in self.net: {net_params_count}")
        
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution
        self.cd_iou_measure = cd_iou_measure
        
        # Create position tensor for query points
        x, y, z = torch.meshgrid(
            torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), 
            torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), 
            torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution)
        )
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)   
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)

        # Load plane points
        plane = np.load("/usr/stud/dira/GraspInClutter/grasping/setup/plane_sampled.npy")
        plane = plane.astype(np.float32)
        self.plane = plane / 0.3 - 0.5

    def __call__(self, state, scene_mesh=None, visual_dict=None, hunyun2_path=None, 
                 scene_name=None, cd_iou_measure=False, target_mesh_gt=None, aff_kwargs={}):
        """
        Perform grasp detection on the given state.
        
        Args:
            state: state object containing scene data and preprocessed complete target data
                   Expected attributes: scene_no_targ_pc, targ_pc, complete_target_pc, complete_target_tsdf
            scene_mesh: scene mesh for visualization (optional)
            visual_dict: dictionary for visualization data (optional)
            hunyun2_path: path for hunyun2 data (unused for ptv3_scene)
            scene_name: scene name (optional)
            cd_iou_measure: whether to measure CD and IoU
            target_mesh_gt: ground truth target mesh (optional, used as fallback)
            aff_kwargs: affordance visualization arguments
            
        Returns:
            grasps, scores, inference_time, cd, iou
        """
        assert state.type == 'ptv3_scene', f"PTV3SceneImplicit only supports ptv3_scene model type, got {state.type}"
        
        visual_dict = {} if visual_dict is None else visual_dict
        print(state.__dict__.keys())

        # Handle ptv3_scene input format
        scene_no_targ_pc = state.scene_no_targ_pc
        scene_no_targ_pc = np.concatenate((scene_no_targ_pc, self.plane), axis=0)

        scene_no_targ_pc =specify_num_points(scene_no_targ_pc, 512)

        complete_targ_pc = state.complete_targ_pc
        complete_targ_pc = specify_num_points(complete_targ_pc, 512)

        full_scene_pc = np.concatenate((scene_no_targ_pc, complete_targ_pc), axis=0)
        scene_labels = np.zeros((len(scene_no_targ_pc), 1), dtype=np.float32)
        target_labels = np.ones((len(complete_targ_pc), 1), dtype=np.float32)
        scene_with_labels = np.concatenate([scene_no_targ_pc, scene_labels], axis=1)
        target_with_labels = np.concatenate([complete_targ_pc, target_labels], axis=1)
        full_scene_with_labels = np.concatenate([scene_with_labels, target_with_labels], axis=0)
        inputs = (full_scene_with_labels, None)

        # save_point_cloud_as_ply(full_scene_with_labels[:, :3], 'full_scene_with_labels.ply')
        
        voxel_size, size = state.tsdf.voxel_size, state.tsdf.size
        
        # Predict using ptv3_scene model with preprocessed complete target data
        with torch.no_grad():
            qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou = predict_ptv3_scene(
                inputs, self.pos, self.net, self.device, 
                visual_dict, state, target_mesh_gt
            )

        begin = time.time()

        # Reshape prediction volumes
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        # Process predictions with completed target grid
        if isinstance(completed_targ_grid, np.ndarray):
            # For numpy array: reshape from (40,40,40) to (1,40,40,40)
            completed_targ_grid = np.expand_dims(completed_targ_grid, axis=0)
        else:
            # For torch tensor: use squeeze and unsqueeze
            completed_targ_grid = completed_targ_grid.squeeze().unsqueeze(0)  # from (40,40,40) to (1,40,40,40)
        
        qual_vol, rot_vol, width_vol = process(completed_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        
        if len(qual_vol.shape) == 1:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
            visual_dict['affordance_visual'] = colored_scene_mesh

        # Select grasps from prediction volumes
        grasps, scores = select(
            qual_vol.copy(), 
            self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), 
            rot_vol, width_vol, 
            threshold=self.qual_th, 
            force_detection=self.force_detection, 
            max_filter_size=8 if self.visualize else 4
        )

        toc = time.time()
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        # Transform grasps to world coordinates
        new_grasps = []
        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        end = time.time()
        print(f"post processing: {end-begin:.3f}s")
        
        return grasps, scores, toc, cd, iou 