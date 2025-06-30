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
import pyvista as pv  
# Import pyrender for rendering
try:
    import pyrender
    from PIL import Image
    PYRENDER_AVAILABLE = True
    print("✓ pyrender package available for rendering")
except ImportError:
    PYRENDER_AVAILABLE = False
    print("✗ pyrender package not available, rendering will be skipped")

LOW_TH = 0.0

# import os
# import numpy as np
# import trimesh
# import pyvista as pv

def render_colored_scene_mesh_with_pyvista(colored_scene_mesh,
                                           output_path="demo/ptv3_scene_affordance_visual.png",
                                           width=640, height=480):
    """
    Render a trimesh object using PyVista with original color, white background, no text or colorbar.
    """
    try:
        if colored_scene_mesh.vertices.size == 0 or colored_scene_mesh.faces.size == 0:
            print("Mesh is empty!")
            return False

        # 获取 mesh 原始颜色
        face_colors = getattr(colored_scene_mesh.visual, "face_colors", None)

        # 转为 PyVista mesh
        faces_flat = np.hstack(
            np.c_[np.full(len(colored_scene_mesh.faces), 3),
                  colored_scene_mesh.faces]
        ).astype(np.int64).ravel()
        pv_mesh = pv.PolyData(colored_scene_mesh.vertices, faces_flat)

        if face_colors is not None:
            pv_mesh.cell_data["colors"] = face_colors
            pv_mesh.cell_data.active_scalars_name = "colors"

        # 创建渲染器
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.add_mesh(
            pv_mesh,
            show_edges=False,
            show_scalar_bar=False  # ✅ 关闭颜色条
        )
        plotter.set_background("white")

        # 设置相机（远视角）
        center = pv_mesh.center
        bounds = pv_mesh.bounds
        extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        iso_dir = np.array([1, 1, 1]) / np.sqrt(3)
        camera_pos = center + iso_dir * extent * 4

        plotter.camera.position = camera_pos.tolist()
        plotter.camera.focal_point = list(center)
        plotter.camera.up = [0, 0, 1]

        # 渲染并保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plotter.screenshot(output_path)
        plotter.close()

        return True

    except Exception as e:
        print(f"Error rendering with PyVista: {e}")
        import traceback; traceback.print_exc()
        return False

    
# def render_colored_scene_mesh_with_pyrender(colored_scene_mesh, output_path="demo/ptv3_scene_affordance_visual.png", 
#                                           width=800, height=600, camera_distance=0.5):
#     """
#     Render colored scene mesh using pyrender package and save it as an image.
    
#     Args:
#         colored_scene_mesh: trimesh object
#         output_path: path to save the rendered image
#         width: image width
#         height: image height
#         camera_distance: distance from camera to scene center
#     """
#     if not PYRENDER_AVAILABLE:
#         print("pyrender not available, skipping rendering")
#         return False
    
#     try:
#         # Ensure the mesh has visual properties
#         if not hasattr(colored_scene_mesh, 'visual') or colored_scene_mesh.visual is None:
#             colored_scene_mesh.visual = trimesh.visual.ColorVisuals()
        
#         # Set all mesh faces to blue color (RGBA) if not already colored
#         if not hasattr(colored_scene_mesh.visual, 'face_colors') or colored_scene_mesh.visual.face_colors is None:
#             colored_scene_mesh.visual.face_colors = np.full((len(colored_scene_mesh.faces), 4), [0, 0, 255, 255], dtype=np.uint8)

#         # Create pyrender scene
#         scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[255, 255, 255])

#         # Add the colored mesh to the scene
#         mesh_render = pyrender.Mesh.from_trimesh(colored_scene_mesh, smooth=False)
#         scene.add(mesh_render)

#         # Set up camera with better positioning
#         camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=float(width) / height)
        
#         # Calculate camera position based on mesh bounds
#         bounds = colored_scene_mesh.bounds
#         center = (bounds[0] + bounds[1]) / 2
#         extent = np.linalg.norm(bounds[1] - bounds[0])
#         camera_distance = max(camera_distance, extent * 1.5)
        
#         camera_pose = np.eye(4)
#         camera_pose[:3, 3] = center + [camera_distance, camera_distance, camera_distance]
#         scene.add(camera, pose=camera_pose)

#         # Set up lighting
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
#         light_pose = np.eye(4)
#         light_pose[:3, 3] = center + [0, -camera_distance, camera_distance]
#         scene.add(light, pose=light_pose)
        
#         # Add ambient light
#         ambient_light = pyrender.AmbientLight(color=[0.5, 0.5, 0.5], intensity=0.5)
#         scene.add(ambient_light)

#         # Render the scene
#         renderer = pyrender.OffscreenRenderer(width, height)
#         color, depth = renderer.render(scene)
        
#         # Clean up
#         renderer.delete()

#         # Ensure output directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # Save the rendered image
#         image = Image.fromarray(color)
#         image.save(output_path)
#         print(f"Rendered image saved to: {output_path}")
#         print(f"Image size: {image.size}, Color range: {color.min()}-{color.max()}")
#         return True

#     except Exception as e:
#         print(f"Error rendering with pyrender: {e}")
#         import traceback
#         traceback.print_exc()
#         return False


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
            
            # Save colored_scene_mesh to demo directory
            # if not os.path.exists('demo'):
            #     os.makedirs('demo')
            
            demo_affordance_path = f"{state.vis_path}/ptv3_scene_affordance_visual.obj"
            colored_scene_mesh.export(demo_affordance_path)
            print(f"Saved affordance visualization to demo: {demo_affordance_path}")
            
            # Render colored scene mesh with pyrender
            render_success = render_colored_scene_mesh_with_pyvista(
                colored_scene_mesh, 
                output_path=f"{state.vis_path}/ptv3_scene_affordance_visual.png",
                width=800, 
                height=600, 
                # camera_distance=0.5
            )
            
            if render_success:
                print("✓ Successfully rendered colored scene mesh with pyrender")
            else:
                print("✗ Failed to render colored scene mesh with pyrender")

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