## TODO target_id, target_body, mesh_pose_dict should be aligne
import os
import argparse
from copy import deepcopy
from pathlib import Path
import random
# import pyvista
import numpy as np
import open3d as o3d
import logging
import json
import uuid
import trimesh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from scipy import ndimage

# Add PyVista for rendering
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    print("Warning: PyVista not available, skipping visualization")
    PYVISTA_AVAILABLE = False

# Add ptvis for rendering
try:
    import ptvis
    PTVIS_AVAILABLE = True
except ImportError:
    print("Warning: ptvis not available, skipping visualization")
    PTVIS_AVAILABLE = False

import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.append('/home/ran.ding/projects/TARGO')
from src.vgn.utils.misc import apply_noise
from src.vgn.io import *
from src.vgn.perception import *
from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils.transform import Rotation, Transform
from src.vgn.utils.implicit import get_mesh_pose_dict_from_world, get_mesh_pose_dict_from_world

MAX_VIEWPOINT_COUNT = 12
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Texture management
TEXTURE_ROOT = Path("/home/ran.ding/projects/TARGO/data/textures")
TEXTURE_CACHE = {}  # Cache for loaded textures


def get_available_textures(texture_root=None):
    """
    Get list of available texture directories.
    
    Parameters:
    - texture_root: Optional custom texture root directory
    
    Returns:
    - List of texture directory names
    """
    if texture_root is None:
        texture_root = TEXTURE_ROOT
    
    if not texture_root.exists():
        print(f"Warning: Texture directory {texture_root} does not exist")
        return []
    
    texture_dirs = []
    for item in texture_root.iterdir():
        if item.is_dir():
            # Check if directory contains texture files
            texture_files = list(item.glob("*.jpg")) + list(item.glob("*.png"))
            if texture_files:
                texture_dirs.append(item.name)
    
    print(f"Found {len(texture_dirs)} texture directories")
    return texture_dirs


def load_texture_material(texture_name, texture_root=None):
    """
    Load a texture material from the textures directory.
    
    Parameters:
    - texture_name: Name of the texture directory
    - texture_root: Optional custom texture root directory
    
    Returns:
    - trimesh.visual.material.Material object or None
    """
    if texture_root is None:
        texture_root = TEXTURE_ROOT
    
    cache_key = f"{texture_root}_{texture_name}"
    if cache_key in TEXTURE_CACHE:
        return TEXTURE_CACHE[cache_key]
    
    texture_dir = texture_root / texture_name
    if not texture_dir.exists():
        print(f"Texture directory does not exist: {texture_dir}")
        return None
    
    # Look for common texture files with correct patterns for AmbientCG textures
    texture_files = {
        'diffuse': (list(texture_dir.glob("*_Color.jpg")) + 
                   list(texture_dir.glob("*_Color.png")) +
                   list(texture_dir.glob("*_Diffuse*.jpg")) + 
                   list(texture_dir.glob("*_Diffuse*.png")) +
                   list(texture_dir.glob("*diffuse*.jpg")) +
                   list(texture_dir.glob("*diffuse*.png")) +
                   list(texture_dir.glob("*.jpg")) +  # Fallback to any jpg
                   list(texture_dir.glob("*.png"))),  # Fallback to any png
        'normal': (list(texture_dir.glob("*_NormalGL.jpg")) +
                  list(texture_dir.glob("*_NormalDX.jpg")) +
                  list(texture_dir.glob("*_Normal*.jpg")) + 
                  list(texture_dir.glob("*_Normal*.png"))),
        'roughness': list(texture_dir.glob("*_Roughness*.jpg")) + list(texture_dir.glob("*_Roughness*.png")),
        'metallic': list(texture_dir.glob("*_Metallic*.jpg")) + list(texture_dir.glob("*_Metallic*.png")),
        'ao': (list(texture_dir.glob("*_AmbientOcclusion.jpg")) +
               list(texture_dir.glob("*_AO*.jpg")) + 
               list(texture_dir.glob("*_AO*.png")))
    }
    
    # Create material
    material = trimesh.visual.material.PBRMaterial()
    
    # Set diffuse texture if available
    if texture_files['diffuse']:
        diffuse_path = str(texture_files['diffuse'][0])
        try:
            # Check if texture file exists and is readable
            if not Path(diffuse_path).exists():
                print(f"Warning: Texture file does not exist: {diffuse_path}")
                return None
            
            # Try to load the image to verify it's valid
            try:
                import PIL.Image
                img = PIL.Image.open(diffuse_path)
                print(f"Successfully loaded image: {diffuse_path} (size: {img.size})")
            except Exception as img_e:
                print(f"Warning: Could not load image {diffuse_path}: {img_e}")
            
            material.baseColorTexture = diffuse_path
            material.baseColorFactor = None  # Let texture colors show
            print(f"Successfully set baseColorTexture: {diffuse_path}")
            
            # Verify the texture was set
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture:
                print(f"Verified: material.baseColorTexture = {material.baseColorTexture}")
            else:
                print("Warning: baseColorTexture was not set properly")
                
        except Exception as e:
            print(f"Warning: Could not load diffuse texture {diffuse_path}: {e}")
            return None
    else:
        print(f"No diffuse texture found in {texture_dir}")
        # List available files for debugging
        all_files = list(texture_dir.glob("*"))
        print(f"Available files in {texture_dir}: {[f.name for f in all_files]}")
        return None
    
    # Set normal texture if available
    if texture_files['normal']:
        normal_path = str(texture_files['normal'][0])
        try:
            material.normalTexture = normal_path
            print(f"Successfully loaded normal texture: {normal_path}")
        except Exception as e:
            print(f"Warning: Could not load normal texture {normal_path}: {e}")
    
    # Set roughness texture if available
    if texture_files['roughness']:
        roughness_path = str(texture_files['roughness'][0])
        try:
            material.roughnessFactor = 1.0
            print(f"Set roughness factor for: {roughness_path}")
        except Exception as e:
            print(f"Warning: Could not set roughness for {roughness_path}: {e}")
    
    # Set metallic texture if available
    if texture_files['metallic']:
        metallic_path = str(texture_files['metallic'][0])
        try:
            material.metallicFactor = 0.5  # Default metallic factor
            print(f"Set metallic factor for: {metallic_path}")
        except Exception as e:
            print(f"Warning: Could not set metallic for {metallic_path}: {e}")
    
    # Cache the material
    TEXTURE_CACHE[cache_key] = material
    return material

def apply_random_texture_to_mesh(mesh, available_textures, texture_root=None):
    """
    Apply a random PBR base-color texture to a Trimesh, ensuring it can be exported to GLB.

    Fixes:
    - Convert material.baseColorTexture from str path to PIL.Image (required by trimesh GLTF exporter).
    - Auto-generate simple UVs if the mesh lacks UVs.
    - Debug-export a GLB (with texture); if it fails, fall back to a no-texture GLB.

    Requirements: Pillow (PIL)
    """
    import random
    from pathlib import Path
    import numpy as np
    from PIL import Image
    import trimesh

    if not available_textures:
        return mesh

    # Pick a texture directory
    texture_name = random.choice(available_textures)
    ## TODO
    from PIL import Image, ImageOps
    import numpy as np
    import trimesh
    from pathlib import Path

    # ---- paths (adjust if needed) ----
    tex_dir = Path("/home/ran.ding/projects/TARGO/data/textures/WoodFloor045")
    p_color = tex_dir / "WoodFloor045_2K-JPG_Color.jpg"
    p_normal = tex_dir / "WoodFloor045_2K-JPG_NormalGL.jpg"     # prefer NormalGL on OpenGL
    p_rough  = tex_dir / "WoodFloor045_2K-JPG_Roughness.jpg"    # wood usually has no metallic map
    # If you had AO/Metallic files, add them here:
    p_ao     = None
    p_metal  = None

    # ---- helpers ----
    def ensure_uv(mesh: trimesh.Trimesh):
        """Return mesh UVs; generate simple planar UVs if missing."""
        has_uv = (
            hasattr(mesh, "visual") and mesh.visual is not None and
            hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and
            len(mesh.visual.uv) == len(mesh.vertices)
        )
        if has_uv:
            return mesh.visual.uv

        vmin = mesh.vertices.min(axis=0)
        vmax = mesh.vertices.max(axis=0)
        span = (vmax - vmin)
        span[span == 0.0] = 1.0

        uv = np.empty((len(mesh.vertices), 2), dtype=np.float32)
        uv[:, 0] = (mesh.vertices[:, 0] - vmin[0]) / span[0]
        uv[:, 1] = (mesh.vertices[:, 1] - vmin[1]) / span[1]
        return uv

    def open_rgba(p):
        return Image.open(p).convert("RGBA") if p else None

    def open_gray(p, fallback_val=255):
        if not p:
            return None
        img = Image.open(p)
        if img.mode != "L":
            img = ImageOps.grayscale(img)
        return img

    # ---- load maps as PIL.Image ----
    img_color = open_rgba(p_color)
    img_normal = open_rgba(p_normal)
    img_rough  = open_gray(p_rough)     # roughness is single-channel
    img_ao     = open_gray(p_ao)        # optional
    img_metal  = open_gray(p_metal)     # optional

    # Basic sanity
    if img_color is None:
        raise FileNotFoundError(f"Base color image not found: {p_color}")

    # ---- build metallicRoughness packed texture (optional but recommended) ----
    # glTF metallic-roughness texture uses:  R=unused (often AO in separate texture), G=roughness, B=metallic
    # If you have AO, glTF uses a separate 'occlusionTexture' (often same image with R=AO).
    def build_mr_and_ao(rough_img, metal_img=None, ao_img=None, size_like=None):
        if rough_img is None and metal_img is None and ao_img is None:
            return None, None
        # choose a base size
        base_size = (rough_img.size if rough_img is not None
                    else metal_img.size if metal_img is not None
                    else ao_img.size if ao_img is not None
                    else (1024, 1024))
        def to_L(img, fill=0):
            if img is None:
                return Image.new("L", base_size, color=fill)
            return ImageOps.grayscale(img).resize(base_size, Image.BILINEAR)

        # channels
        ch_ao    = to_L(ao_img,   fill=255)  # AO default white (no occlusion)
        ch_rough = to_L(rough_img, fill=255) # if missing, assume fully rough
        ch_metal = to_L(metal_img, fill=0)   # if missing, assume non-metal

        # metallicRoughnessTexture (RGB): R(any), G=roughness, B=metallic
        mr_rgb = Image.merge("RGB", (Image.new("L", base_size, 0), ch_rough, ch_metal))
        # occlusionTexture uses R channel (we can reuse the same image)
        ao_rgb = Image.merge("RGB", (ch_ao, Image.new("L", base_size, 0), Image.new("L", base_size, 0)))
        return mr_rgb, ao_rgb

    mr_tex, ao_tex = build_mr_and_ao(img_rough, img_metal, img_ao)

    # ---- attach visuals + PBR material ----
    uv = ensure_uv(mesh)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img_color)

    mat = trimesh.visual.material.PBRMaterial()
    mat.baseColorTexture = img_color
    if img_normal is not None:
        mat.normalTexture = img_normal

    # If we built a packed texture, wire it; also set factors (factors multiply the texture)
    if mr_tex is not None:
        mat.metallicRoughnessTexture = mr_tex
        # wood: non-metal
        mat.metallicFactor = 0.0
        mat.roughnessFactor = 1.0
    else:
        # No packed texture → just use factors
        mat.metallicFactor = 0.0
        mat.roughnessFactor = 0.85  # tweak for wood look

    # AO (optional). If we created a combined image, reuse it.
    if ao_tex is not None:
        mat.occlusionTexture = ao_tex

    # Bind PBR to the mesh visuals
    try:
        mesh.visual.material = mat
    except Exception:
        pass

    # ---- export GLB (images are embedded) ----
    out_glb = "/home/ran.ding/projects/TARGO/tmp_mesh/model_pbr.glb"
    mesh.export(out_glb, file_type="glb")
    print(f"Wrote {out_glb}")

    
    ## TODO



    # mesh = trimesh.load("model.ply", process=False)
    uv = mesh.visual.uv
    im = Image.open("/home/ran.ding/projects/TARGO/data/textures/WoodFloor045/WoodFloor045.png")
    im_normal = Image.open("/home/ran.ding/projects/TARGO/data/textures/WoodFloor045/WoodFloor045_2K-JPG_NormalGL.jpg")
    # im_roughness = Image.open("/home/ran.ding/projects/TARGO/data/textures/WoodFloor045_2K-JPG_Roughness.jpg")
    material = trimesh.visual.texture.SimpleMaterial(image=im, normal_image=im_normal)
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
    mesh.visual = color_visuals
    mesh.export(file_obj='/home/ran.ding/projects/TARGO/tmp_mesh/model03.glb')

    # material = load_texture_material(texture_name, texture_root)
    material = load_texture_material('Metal030', texture_root)
    mesh.visual.material = material
    mesh.export(file_obj='/home/ran.ding/projects/TARGO/tmp_mesh/model04.glb')

    # Helper: ensure we have a PIL.Image (GLTF writer calls .save on it)
    def _to_pil(img_or_path):
        if img_or_path is None:
            return None
        if isinstance(img_or_path, str):
            # Force RGBA for consistency
            return Image.open(img_or_path).convert("RGBA")
        try:
            from PIL.Image import Image as PILImage
            if isinstance(img_or_path, PILImage):
                return img_or_path
        except Exception:
            pass
        return None

    # Normalize the material's baseColorTexture to a PIL image if present
    base_img = None
    if material is not None and hasattr(material, "baseColorTexture"):
        base_img = _to_pil(material.baseColorTexture)
        if base_img is not None:
            material.baseColorTexture = base_img
            # Ensure color comes from the texture (optional)
            if hasattr(material, "baseColorFactor"):
                material.baseColorFactor = None
        else:
            # No valid image: clear the texture so exporter doesn't try to save it
            material.baseColorTexture = None

    # Ensure the mesh has UVs; if not, create simple XY-projected UVs
    def _ensure_uv(mesh_):
        has_uv = (
            hasattr(mesh_, "visual") and
            mesh_.visual is not None and
            hasattr(mesh_.visual, "uv") and
            mesh_.visual.uv is not None and
            len(mesh_.visual.uv) == len(mesh_.vertices)
        )
        if has_uv:
            return mesh_.visual.uv

        vmin = mesh_.vertices.min(axis=0)
        vmax = mesh_.vertices.max(axis=0)
        span = (vmax - vmin)
        span[span == 0.0] = 1.0  # avoid divide-by-zero
        uv = np.zeros((len(mesh_.vertices), 2), dtype=np.float32)
        # Simple XY projection normalized into [0, 1]
        uv[:, 0] = (mesh_.vertices[:, 0] - vmin[0]) / span[0]
        uv[:, 1] = (mesh_.vertices[:, 1] - vmin[1]) / span[1]
        return uv

    # Apply visuals:
    # - If we have a base image, use TextureVisuals + attach PBR material
    # - Otherwise ensure at least a neutral color so viewers don't show it black
    if base_img is not None:
        uv = _ensure_uv(mesh)
        mesh.visual = trimesh.visual.TextureVisuals(mesh=mesh, uv=uv, image=base_img)
        try:
            mesh.visual.material = material  # attach PBR material for GLTF
        except Exception:
            pass
    else:
        # No texture available: ensure colored visuals
        try:
            _ = mesh.visual.face_colors  # if this works, keep existing
        except Exception:
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[200, 200, 200, 255])

    # Debug export: write GLB with texture; on failure, fall back to no-texture GLB
    debug_dir = Path("/home/ran.ding/projects/TARGO/tmp_mesh")
    debug_dir.mkdir(parents=True, exist_ok=True)
    glb_path = debug_dir / f"debug_mesh_{texture_name}.glb"

    try:
        mesh.export(str(glb_path), file_type="glb")
        print(f"[OK] Debug GLB with texture: {glb_path}")
    except Exception as e:
        print(f"[Warn] GLB export with texture failed: {e}")
        # Fallback: drop texture, keep a neutral color to ensure exportability
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[200, 200, 200, 255])
        fallback = debug_dir / f"debug_mesh_{texture_name}_notex.glb"
        try:
            mesh.export(str(fallback), file_type="glb")
            print(f"[OK] Fallback GLB without texture: {fallback}")
        except Exception as e2:
            print(f"[Fail] Fallback GLB export also failed: {e2}")

    return mesh


def visualize_depth_map(depth_img, output_path, scene_id, prefix="depth"):
    """
    Visualize depth map with jet colormap only.
    
    Parameters:
    - depth_img: 2D numpy array containing depth values
    - output_path: Path to save the visualization
    - scene_id: Scene identifier
    - prefix: Prefix for the filename
    """
    try:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Create depth visualization with jet colormap
        plt.imshow(depth_img, cmap='jet')
        plt.colorbar(label='Depth (m)')
        plt.title(f'Depth Map - {scene_id}')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        filename = f"{prefix}_{scene_id}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved depth visualization: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error creating depth visualization: {e}")
        plt.close()
        return False


def visualize_segmentation_map(seg_img, output_path, scene_id, prefix="segmentation"):
    """
    Visualize segmentation map with basic visualization only.
    
    Parameters:
    - seg_img: 2D numpy array containing segmentation IDs
    - output_path: Path to save the visualization
    - scene_id: Scene identifier
    - prefix: Prefix for the filename
    """
    try:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Get unique segment IDs
        unique_ids = np.unique(seg_img)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
        
        # Create custom colormap for segmentation
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ids) + 1))
        colors[0] = [0, 0, 0, 1]  # Black for background
        custom_cmap = ListedColormap(colors)
        
        # Main segmentation visualization
        plt.imshow(seg_img, cmap=custom_cmap, vmin=0, vmax=len(unique_ids))
        plt.colorbar(label='Segment ID')
        plt.title(f'Segmentation Map - {scene_id}')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        filename = f"{prefix}_{scene_id}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved segmentation visualization: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error creating segmentation visualization: {e}")
        plt.close()
        return False


def save_visualization_data(depth_img, seg_img, output_path, scene_id, save_data=True):
    """
    Save depth and segmentation data as numpy arrays for further analysis.
    
    Parameters:
    - depth_img: 2D numpy array containing depth values
    - seg_img: 2D numpy array containing segmentation IDs
    - output_path: Path to save the data
    - scene_id: Scene identifier
    - save_data: Whether to save the raw data arrays
    """
    try:
        if not save_data:
            return True
            
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save depth data
        depth_path = output_path / f"depth_{scene_id}.npy"
        np.save(depth_path, depth_img)
        
        # Save segmentation data
        seg_path = output_path / f"segmentation_{scene_id}.npy"
        np.save(seg_path, seg_img)
        
        # Save metadata
        metadata = {
            'scene_id': scene_id,
            'depth_shape': depth_img.shape,
            'depth_min': float(depth_img.min()),
            'depth_max': float(depth_img.max()),
            'depth_mean': float(depth_img.mean()),
            'seg_shape': seg_img.shape,
            'unique_objects': len(np.unique(seg_img)) - 1,  # Exclude background
            'object_ids': np.unique(seg_img).tolist()
        }
        
        metadata_path = output_path / f"metadata_{scene_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved visualization data: {depth_path}, {seg_path}, {metadata_path}")
        return True
        
    except Exception as e:
        print(f"Error saving visualization data: {e}")
        return False


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


def process_and_store_scene_data(sim, scene_id, target_id, noisy_depth_side_c, seg_side_c, extr_side_c, args):
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

    Returns:
    - clutter_id: Constructed identifier for the clutter data.
    """
    # Generate masks from segmentation data
    mask_targ_side_c = seg_side_c == target_id
    mask_scene_side_c = seg_side_c > 0

    # Generate point clouds for target and scene
    pc_targ_side_c = reconstruct_40_pc(sim, noisy_depth_side_c * mask_targ_side_c, extr_side_c)
    if np.asarray(pc_targ_side_c.points, dtype=np.float32).shape[0] == 0:
        return None
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

    # Save the processed data (occlusion level set to 0.0 since we don't calculate it)
    write_clutter_sensor_data(
        test_root, clutter_id, noisy_depth_side_c, extr_side_c, mask_targ_side_c.astype(int),
        mask_scene_side_c.astype(int), seg_side_c, grid_scene_side_c, grid_targ_side_c,
        pc_scene_depth_side_c, pc_targ_depth_side_c, pc_scene_no_targ_depth_side_c,
        np.asarray(pc_scene_side_c.points, dtype=np.float32),
        np.asarray(pc_targ_side_c.points, dtype=np.float32), pc_scene_no_targ_side_c, 0.0
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
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
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
            
    # if args.is_acronym:
    #     object_count = np.random.randint(2, 6)  # 11 is excluded
    # else:
    object_count = np.random.randint(1, 9)  # 11 is excluded
    sim.reset(object_count, is_gso=True)
    sim.save_state()

    generate_scenes(sim, args)


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
    # Get available textures
    texture_root = args.texture_root if hasattr(args, 'texture_root') else TEXTURE_ROOT
    available_textures = get_available_textures(texture_root)
    if not available_textures:
        print("Warning: No textures available, using original meshes without textures")
    
    depth_side_c, extr_side_c, seg_side_c = render_side_images(sim, 1, random=False, segmentation=True)
    mesh_clutter_pose_dict = get_mesh_pose_dict_from_world(sim.world, sim.object_set, exclude_plane=False)
    ## TODO: add noise to depth image
    # noisy_depth_side_c = np.array([apply_noise(x, args.add_noise) for x in depth_side_c])
    noisy_depth_side_c = depth_side_c
    scene_id = uuid.uuid4().hex
    
    # Create visualization directory only if visualization is enabled
    if args.enable_visualization:
        vis_dir = Path(args.root) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create depth and segmentation visualizations for cluttered scene
        try:
            # Extract first image for visualization (assuming single view)
            depth_img = noisy_depth_side_c[0]
            seg_img = seg_side_c[0]
            
            # Create depth map visualization
            if args.vis_depth:
                visualize_depth_map(depth_img, vis_dir, scene_id, prefix="depth")
            
            # Create segmentation map visualization
            if args.vis_segmentation:
                visualize_segmentation_map(seg_img, vis_dir, scene_id, prefix="segmentation")
            
            # Save raw data for analysis
            if args.save_raw_data:
                save_visualization_data(depth_img, seg_img, vis_dir / "raw_data", scene_id, save_data=True)
            
            logger.info(f"Generated visualizations for scene {scene_id}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations for scene {scene_id}: {e}")
    
    # Save combined mesh file for the entire scene with random textures
    try:
        save_scene_as_combined_mesh_with_textures(sim, scene_id, mesh_clutter_pose_dict, args.root, 
                                                 extrinsics=extr_side_c[0], available_textures=available_textures,
                                                 texture_root=texture_root)
    except Exception as e:
        print(f"Error saving combined scene mesh: {e}")
    
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
    
    # Process only the first visible target instead of all targets
    for target_id in body_ids:
        if count_cluttered[target_id] == 0:  # if the target object is not in the cluttered scene, skip
            continue
        assert target_id != 0 # the plane should not be a target object
        body = target_bodies[target_id]

        #--------------------------------- single scene ---------------------------------##
        target_body = sim.world.load_urdf(body.urdf_path, target_poses[target_id], scale=body.scale)

        # Process and store scene data (without occlusion level calculation)
        clutter_id = process_and_store_scene_data(sim, scene_id, target_id, noisy_depth_side_c, seg_side_c, extr_side_c, args)
        
        if clutter_id is not None:
            write_test_set_point_cloud(args.root, scene_id + f"_c_{target_id}", mesh_clutter_pose_dict, name="mesh_pose_dict")
            
            # Create pyvista rendering for this specific target in the cluttered scene
            output_dir = Path(args.root)
            g1b_dir = output_dir / "g1b_files"
            target_render_dir = g1b_dir / "target_renders"
            create_pyvista_render_with_textures(mesh_clutter_pose_dict, target_render_dir, scene_id, 
                                               target_id=target_id, extrinsics=extr_side_c[0], 
                                               available_textures=available_textures, texture_root=texture_root)
        
        sim.world.remove_body(target_body)
        logger.info(f"scene {scene_id}, target '{target_body.name}' done")
        
        # Only process the first valid target, then break
        break

    logger.info(f"scene {scene_id} done")
    return


def create_pyvista_render_with_textures(mesh_pose_dict, output_path, scene_id, target_id=None, extrinsics=None, available_textures=None, texture_root=None):
    """
    Create PyVista rendering for a scene or single object using extrinsics for camera pose and random textures.
    
    Parameters:
    - mesh_pose_dict: Dictionary containing mesh information and poses
    - output_path: Path to save the rendered images
    - scene_id: Scene identifier
    - target_id: If specified, highlight this object as target
    - extrinsics: Camera extrinsics matrix (4x4) for camera pose
    - available_textures: List of available texture names
    """
    if not PYVISTA_AVAILABLE:
        return False
        
    try:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a combined trimesh scene first
        combined_scene = trimesh.Scene()
        
        # Define colors - red for target, other colors for other objects
        color_map = {
            'red': [255, 0, 0, 255],
            'blue': [0, 0, 255, 255],
            'green': [0, 255, 0, 255],
            'yellow': [255, 255, 0, 255],
            'purple': [128, 0, 128, 255],
            'orange': [255, 165, 0, 255],
            'cyan': [0, 255, 255, 255],
            'magenta': [255, 0, 255, 255]
        }
        colors = ['blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        color_idx = 0
        
        # Add all objects to the combined scene with colors (excluding plane)
        for obj_id, obj_info in mesh_pose_dict.items():
            if obj_id == 0:  # Skip the plane/table
                continue
                
            try:
                # Get mesh file path, scale, and pose from the actual data structure
                mesh_path, scale, pose = obj_info[0], obj_info[1], obj_info[2]
                if not mesh_path or not os.path.exists(mesh_path):
                    continue
                
                # Load mesh
                mesh = trimesh.load(mesh_path)
                if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                    # Apply scale and pose transformation
                    mesh.apply_scale(scale)
                    mesh.apply_transform(pose)
                    
                    # Apply random texture if available
                    if available_textures:
                        mesh = apply_random_texture_to_mesh(mesh, available_textures, texture_root)
                    else:
                        # Fallback to color coding
                        if target_id is not None and obj_id == target_id:
                            color_name = 'red'
                        else:
                            color_name = colors[color_idx % len(colors)]
                            color_idx += 1
                        
                        # Set mesh color
                        mesh.visual.face_colors = color_map[color_name]
                    
                    # Add to combined scene with a unique name
                    combined_scene.add_geometry(mesh, node_name=f"object_{obj_id}")
                    
            except Exception as e:
                print(f"Warning: Could not load mesh for object {obj_id}: {e}")
                continue
        
        # Convert to single mesh for PyVista rendering
        try:
            # Use the new method instead of deprecated dump()
            combined_mesh = combined_scene.to_geometry()
        except Exception as e:
            print(f"Warning: Could not combine scene meshes: {e}")
            return False
        
        if combined_mesh.vertices.size == 0 or combined_mesh.faces.size == 0:
            print("Warning: Combined mesh is empty!")
            return False
        
        # Get mesh face colors
        face_colors = getattr(combined_mesh.visual, "face_colors", None)
        
        # Convert to PyVista mesh
        faces_flat = np.hstack(
            np.c_[np.full(len(combined_mesh.faces), 3),
                  combined_mesh.faces]
        ).astype(np.int64).ravel()
        pv_mesh = pv.PolyData(combined_mesh.vertices, faces_flat)
        
        if face_colors is not None:
            pv_mesh.cell_data["colors"] = face_colors
            pv_mesh.cell_data.active_scalars_name = "colors"
        
        # Render with PyVista
        if extrinsics is not None:
            # Convert 7-element array to 4x4 matrix
            # extrinsics is [x, y, z, qx, qy, qz, qw] format from Transform.to_list()
            if len(extrinsics) == 7:
                # Create Transform object from the 7-element list and get matrix
                from src.vgn.utils.transform import Transform
                transform = Transform.from_list(extrinsics)
                extrinsics_matrix = transform.as_matrix()
            else:
                extrinsics_matrix = extrinsics
            
            # Render with extrinsics-based camera pose
            plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
            plotter.add_mesh(
                pv_mesh,
                show_edges=False,
                show_scalar_bar=False
            )
            plotter.set_background("white")
            
            # Set camera using extrinsics
            # Convert world-to-camera extrinsics to camera-to-world
            camera_to_world = np.linalg.inv(extrinsics_matrix)
            camera_pos = camera_to_world[:3, 3]
            center = pv_mesh.center
            
            # Ensure camera_pos and center are numpy arrays
            camera_pos = np.array(camera_pos)
            center = np.array(center)
            
            plotter.camera.position = camera_pos.tolist()
            plotter.camera.focal_point = center.tolist()
            plotter.camera.up = [0, 0, 1]
            
            if target_id is not None:
                filename = f"{scene_id}_target_{target_id}_extrinsics.png"
            else:
                filename = f"{scene_id}_combined_extrinsics.png"
            
            render_path = output_path / filename
            plotter.screenshot(str(render_path))
            plotter.close()
            print(f"Rendered with extrinsics: {render_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating PyVista render: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_scene_as_combined_mesh_with_textures(sim, scene_id, mesh_pose_dict, output_dir, extrinsics=None, available_textures=None, texture_root=None):
    """
    Save all objects in the scene as a GLB file with multiple objects as separate nodes and random textures.
    
    Parameters:
    - sim: Simulation instance
    - scene_id: Scene identifier
    - mesh_pose_dict: Dictionary containing mesh information and poses
    - output_dir: Output directory path
    - extrinsics: Camera extrinsics matrix (4x4) for camera pose
    - available_textures: List of available texture names
    """
    try:
        output_dir = Path(output_dir)
        g1b_dir = output_dir / "g1b_files"
        g1b_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a trimesh scene to preserve multiple objects as separate nodes
        scene = trimesh.Scene()
        
        # Track applied textures for metadata
        applied_textures = {}
        
        # Add all objects from the scene as separate nodes (excluding plane)
        print(f"Processing {len(mesh_pose_dict)} objects in mesh_pose_dict")
        objects_added = 0
        
        for obj_id, obj_info in mesh_pose_dict.items():
            if obj_id == 0:  # Skip the plane/table
                continue
            
            try:
                mesh_path, scale, pose = obj_info[0], obj_info[1], obj_info[2]
                print(f"Processing object {obj_id}: {mesh_path}")
                
                # Load mesh with original textures
                mesh = trimesh.load(mesh_path, process=False)  # process=False to preserve original textures
                
                # Handle different mesh types (single mesh or scene with multiple meshes)
                if isinstance(mesh, trimesh.Scene):
                    # If it's a scene, extract all meshes and add them individually
                    for mesh_name, mesh_obj in mesh.geometry.items():
                        if hasattr(mesh_obj, 'vertices'):  # Ensure it's a mesh
                            mesh_obj.apply_scale(scale)
                            mesh_obj.apply_transform(pose)
                            
                            # Apply random texture if available
                            if available_textures:
                                mesh_obj = apply_random_texture_to_mesh(mesh_obj, available_textures, texture_root)
                                # Track which texture was applied
                                if hasattr(mesh_obj.visual, 'material') and hasattr(mesh_obj.visual.material, 'baseColorTexture'):
                                    texture_path = mesh_obj.visual.material.baseColorTexture
                                    texture_name = Path(texture_path).parent.name if texture_path else "none"
                                    applied_textures[f"{obj_id}_{mesh_name}"] = texture_name
                            
                            # Add as separate node in scene
                            scene.add_geometry(mesh_obj, node_name=f"object_{obj_id}_{mesh_name}")
                else:
                    # Single mesh
                    mesh.apply_scale(scale)
                    mesh.apply_transform(pose)
                    
                    # Apply random texture if available
                    if available_textures:
                        mesh = apply_random_texture_to_mesh(mesh, available_textures, texture_root)
                        # Track which texture was applied
                        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'baseColorTexture'):
                            texture_path = mesh.visual.material.baseColorTexture
                            texture_name = Path(texture_path).parent.name if texture_path else "none"
                            applied_textures[str(obj_id)] = texture_name
                    
                    # Add as separate node in scene
                    scene.add_geometry(mesh, node_name=f"object_{obj_id}")
                    objects_added += 1
                    print(f"Successfully added object {obj_id} to scene")
                    
            except Exception as e:
                print(f"Warning: Could not add object {obj_id} to scene: {e}")
                continue
        
        print(f"Total objects added to scene: {objects_added}")
        
        # Save the scene as GLB file (preserving multiple objects as separate nodes)
        base_filename = f"{scene_id}_combined"
        glb_path = g1b_dir / f"{base_filename}.glb"
        
        try:
            # Check if scene has any geometry
            if len(scene.geometry) == 0:
                print("Warning: Scene has no geometry, cannot export GLB")
                return False
            
            # Export scene as GLB - this preserves the multi-object structure
            glb_data = scene.export(file_type='glb')
            with open(glb_path, 'wb') as f:
                f.write(glb_data)
            print(f"Saved GLB with multiple objects as separate nodes: {glb_path}")
            print(f"Scene contains {len(scene.geometry)} objects as separate nodes")
            
        except Exception as e:
            print(f"Error saving GLB file: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Save metadata as JSON (excluding plane)
        metadata_path = g1b_dir / f"{base_filename}_metadata.json"
        metadata = {
            'scene_id': scene_id,
            'object_count': len([k for k in mesh_pose_dict.keys() if k != 0]),  # Exclude plane
            'objects': {str(k): str(v) for k, v in mesh_pose_dict.items() if k != 0},
            'applied_textures': applied_textures,
            'textures_available': len(available_textures) if available_textures else 0,
            'scene_nodes': list(scene.graph.nodes.keys()),
            'file_format': 'GLB',
            'preserves_multi_objects': True
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
        # Create ptvis rendering using extrinsics
        render_dir = g1b_dir / "renders"
        create_pyvista_render_with_textures(mesh_pose_dict, render_dir, scene_id, 
                                           extrinsics=extrinsics, available_textures=available_textures, texture_root=texture_root)
        
        return True
        
    except Exception as e:
        print(f"Error saving scene as GLB: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path, default= '/home/ran.ding/projects/TARGO/messy_kitchen_scenes/gso_pile_scenes_test_2')
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="mess_kitchen/test", choices=["mess_kitchen/train",  "mess_kitchen/test"])
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-scenes", type=int, default=100, help="Number of scenes to generate")
    parser.add_argument("--save-scene", default=True)
    parser.add_argument("--random", action="store_true", help="Add distribution to camera pose")
    parser.add_argument("--sim-gui", action="store_true", default=False)
    parser.add_argument("--add-noise", type=str, default='norm', help = "norm_0.005 | norm | dex")
    parser.add_argument("--num-proc", type=int, default=2, help="Number of processes to use")
    parser.add_argument("--is-acronym", action="store_true", default=True)
    parser.add_argument("--is-ycb", action="store_true", default=False)
    parser.add_argument("--is-egad", action="store_true", default=False)
    parser.add_argument("--is-g1b", action="store_true", default=False)
    
    # Visualization options - enabled by default for easier use
    parser.add_argument("--enable-visualization", action="store_true", default=True, help="Enable visualization of scenes (default: True)")
    parser.add_argument("--vis-depth", action="store_true", default=True, help="Visualize depth maps (default: True)")
    parser.add_argument("--vis-segmentation", action="store_true", default=True, help="Visualize segmentation maps (default: True)")
    parser.add_argument("--save-raw-data", action="store_true", default=True, help="Save raw data for analysis (default: True)")
    parser.add_argument("--vis-all", action="store_true", default=False, help="Enable all visualization options")
    parser.add_argument("--no-visualization", action="store_true", default=False, help="Disable all visualization")
    
    # Texture options
    parser.add_argument("--texture-root", type=Path, default=TEXTURE_ROOT, help="Root directory for textures")
    parser.add_argument("--no-textures", action="store_true", default=False, help="Disable texture application")

    args = parser.parse_args()
    
    # Update texture root if specified
    if args.texture_root != TEXTURE_ROOT:
        print(f"Using custom texture root: {args.texture_root}")
        print(f"Default texture root: {TEXTURE_ROOT}")
    
    # If --no-visualization is specified, disable all visualization
    if args.no_visualization:
        args.enable_visualization = False
        args.vis_depth = False
        args.vis_segmentation = False
        args.save_raw_data = False
    
    # If --vis-all is specified, enable all visualization options
    if args.vis_all:
        args.enable_visualization = True
        args.vis_depth = True
        args.vis_segmentation = True
        args.save_raw_data = True
    
    print("="*60)
    print("Scene Generation with Random Textures")
    print("="*60)
    print(f"Output directory: {args.root}")
    print(f"Number of scenes: {args.num_scenes}")
    print(f"Texture root: {TEXTURE_ROOT}")
    print(f"Textures enabled: {not args.no_textures}")
    print(f"Visualization enabled: {args.enable_visualization}")
    if args.enable_visualization:
        print(f"  - Depth visualization: {args.vis_depth}")
        print(f"  - Segmentation visualization: {args.vis_segmentation}")
        print(f"  - Save raw data: {args.save_raw_data}")
    print("="*60)
    
    # Generate specified number of scenes
    for i in range(args.num_scenes):
        logger.info(f"Generating scene {i+1}/{args.num_scenes}")
        main(args)
    
    logger.info(f"Generated {args.num_scenes} scenes successfully")
    
    if args.enable_visualization:
        print("\n" + "="*60)
        print("Visualization Summary")
        print("="*60)
        print(f"Visualization files saved to: {args.root}/visualizations/")
        print(f"Raw data saved to: {args.root}/visualizations/raw_data/")
        print("="*60)
