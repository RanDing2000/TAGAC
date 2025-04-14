import numpy as np
import open3d as o3d

def align_to_depth_hunyuan3d_v3(mesh, pts3d):
    # Define the rotations to be tested
    rotations = [
        np.eye(3),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
        np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]]),
        np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    ]

    best_fitness = float('inf')
    best_mesh = None
    best_scale = None

    for rot_idx, rot_matrix in enumerate(rotations):
        # Create a copy of the mesh for this rotation
        rotated_mesh = mesh.copy()
        
        # Apply rotation
        rotation_transform = np.eye(4)
        rotation_transform[:3, :3] = rot_matrix
        rotated_mesh.apply_transform(rotation_transform)
        
        # Calculate initial scale based on bounding boxes
        bbx_src = rotated_mesh.bounds
        src_extents = bbx_src[1] - bbx_src[0]
        target_extents = bbx_target.get_extent()
        
        max_src_extent = np.max(src_extents)
        max_target_extent = np.max(target_extents)
        initial_scale = max_target_extent / max_src_extent
        
        # Apply initial scale
        rotated_mesh.apply_scale(initial_scale)
        
        # Store initial scale for later use with rotation scale
        current_initial_scale = initial_scale
        
        # Center alignment
        bbx_src = rotated_mesh.bounds
        initial_shift = np.mean(bbx_src, axis=0) - bbx_target.get_center()
        rotated_mesh.vertices = rotated_mesh.vertices - initial_shift
        
        # ... existing code ...

        # Update best result across all rotations
        if rotation_best_fitness > best_fitness:
            best_fitness = rotation_best_fitness
            transformed_mesh = rotated_mesh.copy()
            
            # Apply the best scale directly after initial scale (as combined scale)
            # First reset to original position
            mesh_center = transformed_mesh.centroid
            transformed_mesh.apply_translation(-mesh_center)
            
            # Apply the combined scale (initial_scale was already applied, now apply rotation_best_scale)
            transformed_mesh.apply_scale(rotation_best_scale)
            
            transformed_mesh.apply_translation(mesh_center)
            
            # Now apply the transformation
            transformed_mesh.apply_transform(np.linalg.inv(rotation_best_transform))
            best_mesh = transformed_mesh
            
            # Track combined scale for reporting
            best_scale = current_initial_scale * rotation_best_scale
            
            # Debug output for best rotation so far
            print(f"Rotation {rot_idx} gave best fitness so far: {rotation_best_fitness:.4f} with combined scale {best_scale:.3f} (initial: {current_initial_scale:.3f}, rotation: {rotation_best_scale:.3f})")
            source_pc_aligned = best_mesh.sample(len(pts3d))
            source_pc_aligned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pc_aligned))
            o3d.io.write_point_cloud(f"source_aligned_rot_{rot_idx}.ply", source_pc_aligned)

    return best_mesh, best_scale 