import numpy as np
from scipy.spatial.transform import Rotation
from vgn.grasp import Grasp
from vgn.utils.transform import Transform

def anygrasp_to_vgn(grasp_group, extrinsic, workspace_size=None, gripper_offset=np.array([0, 0, -0.02])):
    """
    Convert grasps from AnyGrasp/GraspNet format to VGN format
    
    Args:
        grasp_group: GraspGroup from AnyGrasp/GraspNet
        extrinsic: Camera extrinsic matrix (camera to world transform)
        workspace_size: Size limits of the workspace for filtering [optional]
        gripper_offset: Offset to apply for the Franka gripper [default: 2cm in -Z direction]
    
    Returns:
        vgn_grasps: List of VGN Grasp objects
        scores: List of corresponding grasp scores
    """
    vgn_grasps = []
    scores = []
    
    for grasp in grasp_group:
        # Add rotation by Y axis 90 degrees to convert to VGN-style grasps
        grasp_R = Rotation.from_matrix(grasp.rotation_matrix) * Rotation.from_euler('Y', np.pi/2)
        grasp_tf = Transform(grasp_R, grasp.translation)
        
        # Transform to world frame
        grasp_tf_world = np.linalg.inv(extrinsic.as_matrix()) @ grasp_tf.as_matrix()
        
        # Filter grasps outside workspace if workspace size is provided
        if workspace_size is not None:
            if (grasp_tf_world[0,3] > workspace_size or grasp_tf_world[0,3] < 0 or 
                grasp_tf_world[1,3] > workspace_size or grasp_tf_world[1,3] < 0):
                continue
        
        # Add offset for Franka gripper
        if gripper_offset is not None:
            offset_tf = Transform(Rotation.from_matrix(np.eye(3)), gripper_offset)
            grasp_tf_world = grasp_tf_world @ offset_tf.as_matrix()

        # Create VGN grasp from the transformed pose
        curr_grasp = Grasp(Transform.from_matrix(grasp_tf_world), grasp.width)
        vgn_grasps.append(curr_grasp)
        scores.append(grasp.score)
    
    return vgn_grasps, scores

def fgc_to_vgn(grasp_group, extrinsic, workspace_size=None, gripper_offset=np.array([0, 0, -0.02])):
    """
    Convert grasps from FGC-Grasp format to VGN format.
    This uses the same conversion as AnyGrasp as they have similar formats.
    
    Args:
        grasp_group: GraspGroup from FGC-Grasp
        extrinsic: Camera extrinsic matrix (camera to world transform)
        workspace_size: Size limits of the workspace for filtering [optional]
        gripper_offset: Offset to apply for the Franka gripper [default: 2cm in -Z direction]
    
    Returns:
        vgn_grasps: List of VGN Grasp objects
        scores: List of corresponding grasp scores
    """
    # FGC-Grasp uses the same format as AnyGrasp for grasp representation
    return anygrasp_to_vgn(grasp_group, extrinsic, workspace_size, gripper_offset)

def vgn_to_anygrasp(vgn_grasps, scores, extrinsic):
    """
    Convert grasps from VGN format to AnyGrasp/GraspNet format
    
    Args:
        vgn_grasps: List of VGN Grasp objects
        scores: List of grasp scores
        extrinsic: Camera extrinsic matrix (world to camera transform)
    
    Returns:
        grasp_group: GraspGroup for AnyGrasp/GraspNet
    
    Note: 
        This function requires graspnetAPI's GraspGroup class to be imported.
        Import it only when this function is called to avoid dependency issues.
    """
    try:
        from graspnetAPI.grasp import GraspGroup, Grasp as GraspnetGrasp
    except ImportError:
        raise ImportError("graspnetAPI is required for vgn_to_anygrasp conversion. "
                         "Please install it with: pip install graspnetAPI")
    
    grasp_list = []
    
    for i, vgn_grasp in enumerate(vgn_grasps):
        # Get the grasp pose in world frame
        grasp_tf_world = vgn_grasp.pose.as_matrix()
        
        # Remove the gripper offset
        offset = np.array([0, 0, 0.02])  # Reverse the offset applied in anygrasp_to_vgn
        offset_tf = Transform(Rotation.from_matrix(np.eye(3)), offset)
        grasp_tf_world = grasp_tf_world @ offset_tf.as_matrix()
        
        # Transform to camera frame
        grasp_tf_camera = extrinsic.as_matrix() @ grasp_tf_world
        
        # Remove the Y-axis 90 degree rotation
        grasp_R = Rotation.from_matrix(grasp_tf_camera[:3, :3]) * Rotation.from_euler('Y', -np.pi/2)
        
        # Create AnyGrasp/GraspNet grasp
        score = scores[i] if i < len(scores) else 0.0
        grasp = GraspnetGrasp(
            rotation_matrix=grasp_R.as_matrix(),
            translation=grasp_tf_camera[:3, 3],
            width=vgn_grasp.width,
            score=score
        )
        grasp_list.append(grasp)
    
    return GraspGroup(grasp_list) 