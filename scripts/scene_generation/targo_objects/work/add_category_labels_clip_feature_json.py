import os
import numpy as np
import json
import torch
import clip
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import trimesh
from collections import defaultdict
from src.vgn.utils.transform import Rotation, Transform

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_clip_model():
    """
    Load CLIP model for feature extraction.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_targo_category_mapping():
    """
    Load TARGO category mapping from files.
    """
    # Load class names with indices
    class_names_path = Path("data/targo_category/class_names.json")
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    # Load object to category mapping
    vgn_objects_path = Path("data/targo_category/vgn_objects_category.json")
    with open(vgn_objects_path, 'r') as f:
        object_category_mapping = json.load(f)
    
    return class_names, object_category_mapping

def extract_object_category_from_path(mesh_path, object_category_mapping):
    """
    Extract object category from mesh file path using TARGO mapping.
    
    Args:
        mesh_path: Path to mesh file
        object_category_mapping: Mapping from object names to categories
        
    Returns:
        category: Extracted category name
    """
    # Extract filename without extension
    filename = os.path.basename(mesh_path).replace("_textured.obj", "").replace(".obj", "")
    
    # Try to get category from TARGO mapping
    if filename in object_category_mapping:
        return object_category_mapping[filename]
    
    # Fallback: try to extract category from filename
    parts = filename.split("_")
    if len(parts) > 1:
        potential_category = parts[0].lower()
        return potential_category
    
    return "others"

def get_clip_features_for_categories(categories, model, device):
    """
    Pre-compute CLIP features for all categories.
    
    Args:
        categories: List of category names
        model: CLIP model
        device: Device to run CLIP on
        
    Returns:
        category_features: Dictionary mapping category to CLIP features
    """
    category_features = {}
    
    for category in categories:
        text_inputs = torch.cat([clip.tokenize(f"a {category}")]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            category_features[category] = text_features.cpu().numpy().flatten()
    
    return category_features

def build_pybullet_scene(sim, mesh_pose_dict):
    """
    Step 1: Build pybullet scene from mesh_pose_dict
    
    Args:
        sim: Simulation instance
        mesh_pose_dict: Dictionary containing mesh pose information
        
    Returns:
        object_meshes: List of loaded object meshes
        object_poses: List of object poses
        object_scales: List of object scales
        object_categories: List of object categories
    """
    print("Step 1: Building pybullet scene...")
    
    # Prepare simulator
    sim.world.reset()
    sim.world.set_gravity([0.0, 0.0, -9.81])
    sim.draw_workspace()
    sim.save_state()

    # Manually adjust boundaries
    sim.lower = np.array([0.02, 0.02, 0.055])
    sim.upper = np.array([0.28, 0.28, 0.30000000000000004])

    object_meshes = []
    object_poses = []
    object_scales = []
    object_categories = []
    
    # Load TARGO category mapping
    class_names, object_category_mapping = load_targo_category_mapping()
    
    # Place objects in pybullet scene
    for obj_id, (mesh_path, scale, pose) in enumerate(mesh_pose_dict.values()):
        # Load mesh for trimesh processing
        mesh = trimesh.load(mesh_path)
        object_meshes.append(mesh)
        object_poses.append(pose)
        object_scales.append(scale)
        
        # Extract category
        mesh_filename = mesh_path.split("/")[-1]
        if mesh_filename == 'plane.obj':
            mesh_id = mesh_filename.split("_")[0].split(".")[0]
        else:
            mesh_id = mesh_filename.replace('.obj', '')
            if mesh_id.endswith('_visual'):
                mesh_id = mesh_id[:-7]
        
        category = extract_object_category_from_path(mesh_id, object_category_mapping)
        object_categories.append(category)
        
        # Load object in pybullet
        pose_transform = Transform.from_matrix(pose)
        
        # Handle different file types
        file_basename = os.path.basename(mesh_path)
        if file_basename == 'plane.obj':
            urdf_path = mesh_path.replace(".obj", ".urdf")
        else:
            file_id = file_basename.replace("_textured.obj", "").replace(".obj", "")
            urdf_base_dir = "/home/ran.ding/projects/TARGO/data//urdfs/packed/train"
            urdf_path = f"{urdf_base_dir}/{file_id}.urdf"
            
            # If not found, try with category prefix
            if not os.path.exists(urdf_path):
                import glob
                matching_files = glob.glob(f"{urdf_base_dir}/*_{file_id}.urdf")
                if matching_files:
                    urdf_path = matching_files[0]
        
        # Load URDF in pybullet
        body = sim.world.load_urdf(
            urdf_path=urdf_path,
            pose=pose_transform,
            scale=scale
        )
        
        print(f"Loaded object {obj_id}: {mesh_id} -> {category}")
    
    print(f"Pybullet scene built with {len(object_meshes)} objects")
    return object_meshes, object_poses, object_scales, object_categories

def get_category_labels(points, object_meshes, object_poses, object_scales, object_categories, target_id, class_names):
    """
    Step 2: Get category labels for points
    
    Args:
        points: Point cloud array [N, 3]
        object_meshes: List of object meshes
        object_poses: List of object poses
        object_scales: List of object scales
        object_categories: List of object categories
        target_id: Target object ID
        class_names: Mapping from category names to indices
        
    Returns:
        object_labels: Object labels for each point [N]
        instance_labels: Instance labels (0 for scene, 1 for target) [N]
        category_labels: Category labels for each point [N]
    """
    print("Step 2: Getting category labels...")
    
    N = points.shape[0]
    object_labels = np.zeros(N, dtype=np.int32)
    instance_labels = np.zeros(N, dtype=np.int32)
    category_labels = np.zeros(N, dtype=np.int32)
    
    # Convert points to world coordinates
    world_points = (points + 0.5) * 0.3  # Convert from [-0.5, 0.5] to world coordinates
    
    for obj_id, (mesh, pose, scale, category) in enumerate(zip(object_meshes, object_poses, object_scales, object_categories)):
        # Transform mesh to world coordinates
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_scale(scale)
        transformed_mesh.apply_transform(pose)
        
        # Check which points are inside the object's bounding box
        bbox_min = transformed_mesh.bounds[0]
        bbox_max = transformed_mesh.bounds[1]
        
        # Add some tolerance
        tolerance = 0.01
        bbox_min -= tolerance
        bbox_max += tolerance
        
        # Find points inside bounding box
        inside_mask = np.all((world_points >= bbox_min) & (world_points <= bbox_max), axis=1)
        
        # Assign object label
        object_labels[inside_mask] = obj_id
        
        # Assign instance label (1 for target, 0 for others)
        if obj_id == target_id:
            instance_labels[inside_mask] = 1
        
        # Assign category label
        category_idx = class_names.get(category, class_names["others"])
        category_labels[inside_mask] = category_idx
    
    print(f"Category labels assigned: {len(np.unique(category_labels))} unique categories")
    return object_labels, instance_labels, category_labels

def get_category_clip_features(category_labels, class_names, clip_features_dict):
    """
    Step 3: Get CLIP features for each point based on category labels
    
    Args:
        category_labels: Category labels for each point [N]
        class_names: Mapping from category names to indices
        clip_features_dict: Pre-computed CLIP features for categories
        
    Returns:
        clip_features: CLIP features for each point [N, 512]
    """
    print("Step 3: Getting CLIP features...")
    
    N = len(category_labels)
    clip_features = np.zeros((N, 512), dtype=np.float32)
    
    for i, cat_idx in enumerate(category_labels):
        # Find category name from index
        category_name = None
        for cat_name, idx in class_names.items():
            if idx == cat_idx:
                category_name = cat_name
                break
        
        if category_name and category_name in clip_features_dict:
            clip_features[i] = clip_features_dict[category_name]
    
    print(f"CLIP features computed: {clip_features.shape}")
    return clip_features

def process_npz_file_with_steps(npz_path, mesh_pose_path, model, preprocess, device, class_names, 
                               object_category_mapping, clip_features_dict, output_dir):
    """
    Process a single npz file using the three-step approach.
    
    Args:
        npz_path: Path to the npz file
        mesh_pose_path: Path to corresponding mesh_pose_dict file
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Device to run CLIP on
        class_names: Mapping from category names to indices
        object_category_mapping: Mapping from object names to categories
        clip_features_dict: Pre-computed CLIP features for categories
        output_dir: Output directory for processed files
    """
    # try:
    print(f"\nProcessing {npz_path.name}...")
    scene_name = npz_path.stem
    target_id = int(scene_name.split('_')[-1])  # Convert to 0-based index
    
    # Load the npz file
    data = np.load(npz_path, allow_pickle=True)
    
    # Check if mask_targ exists in data
    if 'mask_targ' not in data:
        print(f"Warning: mask_targ not found in {scene_name}")
        return None
    
    # Load mesh_pose_dict
    mesh_pose_data = np.load(mesh_pose_path, allow_pickle=True)
    mesh_pose_dict = mesh_pose_data["pc"].item()
    if '_c_' in scene_name:
        segmentation_map = data['segmentation_map']
        mask_targ = data['mask_targ']
    elif '_s_' in scene_name:
        # segmentation_map = data['segmentation_map']
        mask_targ = data['mask_targ']
    
    # if '_c_' in scene_name:
    object_meshes = []
    obj_id_list = np.unique(list(mesh_pose_dict.keys()))
    object_poses = []
    object_scales = []
    object_categories_dict = {}
    
    for obj_id, (mesh_path, scale, pose) in enumerate(mesh_pose_dict.values()):
        # Load mesh
        mesh = trimesh.load(mesh_path)
        object_meshes.append(mesh)
        object_poses.append(pose)
        object_scales.append(scale)
        mesh_filename = mesh_path.split("/")[-1]
        if mesh_filename == 'plane.obj':
            mesh_id = mesh_filename.split("_")[0].split(".")[0]
        else:
            mesh_id = mesh_filename.replace('.obj', '')
            # For files like 'PineappleSlices_800_tex_visual.obj', remove the trailing '_visual'
            if mesh_id.endswith('_visual'):
                mesh_id = mesh_id[:-7]
        # Extract category using TARGO mapping
        category = extract_object_category_from_path(mesh_id, object_category_mapping)
        object_categories_dict[obj_id_list[obj_id]] = category

    if '_c_' in scene_name:
        scene_name = npz_path.stem  # '6776236d6eed443ebec7405633aadbdd_c_2'
        assert np.any((segmentation_map == target_id) == (mask_targ == True)) == True
        target_category = object_categories_dict[target_id]
    elif '_s_' in scene_name:
        assert len(object_categories_dict) == 2
        target_category = object_categories_dict[obj_id_list[-1]]
    
    return target_category
        

    # except Exception as e:
    #     print(f"Error processing {npz_path}: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Add category labels and CLIP features to scene npz files")
    parser.add_argument("--scenes_dir", type=str, 
                       default="data_scenes/targo_dataset/scenes",
                       help="Directory containing scene npz files")
    parser.add_argument("--mesh_pose_dir", type=str,
                       default="data_scenes/targo_dataset/mesh_pose_dict",
                       help="Directory containing mesh_pose_dict files")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Use the same scenes directory for output
    scenes_dir = Path(args.scenes_dir)
    mesh_pose_dir = Path(args.mesh_pose_dir)
    
    # Create tmp directory if it doesn't exist
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model()
    
    # Load TARGO category mappings
    print("Loading TARGO category mappings...")
    class_names, object_category_mapping = load_targo_category_mapping()
    categories = list(class_names.keys())
    
    print(f"CLIP model loaded on {device}")
    print(f"Processing {len(categories)} categories")
    
    # Pre-compute CLIP features for all categories
    print("Pre-computing CLIP features for categories...")
    clip_features_dict = get_clip_features_for_categories(categories, model, device)
    
    # Get all npz files
    npz_files = list(scenes_dir.glob("*.npz"))
    
    if args.max_files:
        npz_files = npz_files[:args.max_files]
    
    print(f"Found {len(npz_files)} npz files to process")

    category_scene_dict = {}
    no_mask_targ_files = []
    # start_flag = False
    # Process each npz file using three-step approach
    for npz_file in tqdm(npz_files, desc="Processing npz files"):
        scene_name = npz_file.stem
        if scene_name == 'b960209b0cbd406d98dac25aeccd3c71_s_2':
            continue
        # start_flag = True
        # if start_flag == False:
        #     continue

        # Find corresponding mesh_pose_dict file
        import re
        match = re.match(r"(.+_c)(?:_\d+)?\.npz", npz_file.name)
        if match:
            mesh_pose_filename = f"{match.group(1)}.npz"
        else:
            mesh_pose_filename = npz_file.name
        mesh_pose_file = mesh_pose_dir / mesh_pose_filename

        if mesh_pose_file.exists():
            target_category = process_npz_file_with_steps(npz_file, mesh_pose_file, model, preprocess, device, 
                                      class_names, object_category_mapping, clip_features_dict, scenes_dir)
            if target_category is None:
                # Record file without mask_targ
                no_mask_targ_files.append(scene_name)
                continue
            category_scene_dict[scene_name] = target_category
        else:
            print(f"Warning: No mesh_pose_dict file found for {npz_file.name}")
    
    # Save files without mask_targ to tmp/no_mask_targ_file.txt
    if no_mask_targ_files:
        no_mask_targ_file_path = tmp_dir / "no_mask_targ_file.txt"
        with open(no_mask_targ_file_path, 'w') as f:
            for scene_name in no_mask_targ_files:
                f.write(f"{scene_name}\n")
        print(f"Found {len(no_mask_targ_files)} files without mask_targ, saved to {no_mask_targ_file_path}")
    else:
        print("All processed files have mask_targ data")
    
    with open("data_scenes/targo_dataset/category_scene_dict.json", "w") as f:
        json.dump(category_scene_dict, f, indent=2)
    
if __name__ == "__main__":    
    main()
