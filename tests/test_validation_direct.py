#!/usr/bin/env python3
"""
Direct test script for validation function to debug wandb issues.
"""

import torch
import tempfile
import os
from pathlib import Path
import sys
import numpy as np

# Add the project root to Python path
project_root = "/usr/stud/dira/GraspInClutter/targo"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✓ wandb is available")
except ImportError:
    WANDB_AVAILABLE = False
    print("✗ wandb is not available")
    exit(1)

def test_direct_validation():
    """Test validation function directly."""
    
    print("[DEBUG] Testing direct validation function...")
    
    # Initialize wandb
    wandb.init(
        project="test-direct-validation",
        name="test-direct-validation-run",
        config={"test": True}
    )
    
    try:
        from src.vgn.detection_implicit import VGNImplicit
        from src.vgn.experiments import target_sample_offline_ycb, target_sample_offline_acronym
        
        print("[DEBUG] Successfully imported validation modules")
        
        # Create a dummy model file for testing
        temp_model_path = tempfile.mktemp(suffix='.pt')
        dummy_model_data = {'test': 'data'}
        torch.save(dummy_model_data, temp_model_path)
        print(f"[DEBUG] Created dummy model file: {temp_model_path}")
        
        # Test paths
        ycb_test_root = 'data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000'
        acronym_test_root = 'data_scenes/acronym/acronym-slight-occlusion-1000'
        
        print(f"[DEBUG] Checking YCB test root: {ycb_test_root}")
        print(f"[DEBUG] YCB path exists: {os.path.exists(ycb_test_root)}")
        print(f"[DEBUG] Checking ACRONYM test root: {acronym_test_root}")
        print(f"[DEBUG] ACRONYM path exists: {os.path.exists(acronym_test_root)}")
        
        # Check required files
        ycb_occ_path = f'{ycb_test_root}/test_set/occlusion_level_dict.json'
        acronym_occ_path = f'{acronym_test_root}/test_set/occlusion_level_dict.json'
        
        print(f"[DEBUG] YCB occlusion dict exists: {os.path.exists(ycb_occ_path)}")
        print(f"[DEBUG] ACRONYM occlusion dict exists: {os.path.exists(acronym_occ_path)}")
        
        # Check scenes directory
        ycb_scenes = f'{ycb_test_root}/scenes'
        acronym_scenes = f'{acronym_test_root}/scenes'
        
        print(f"[DEBUG] YCB scenes exists: {os.path.exists(ycb_scenes)}")
        print(f"[DEBUG] ACRONYM scenes exists: {os.path.exists(acronym_scenes)}")
        
        if os.path.exists(ycb_scenes):
            print(f"[DEBUG] YCB scenes count: {len(os.listdir(ycb_scenes))}")
        if os.path.exists(acronym_scenes):
            print(f"[DEBUG] ACRONYM scenes count: {len(os.listdir(acronym_scenes))}")
            
        # Check mesh_pose_dict directory
        ycb_mesh_pose = f'{ycb_test_root}/mesh_pose_dict'
        acronym_mesh_pose = f'{acronym_test_root}/mesh_pose_dict'
        
        print(f"[DEBUG] YCB mesh_pose_dict exists: {os.path.exists(ycb_mesh_pose)}")
        print(f"[DEBUG] ACRONYM mesh_pose_dict exists: {os.path.exists(acronym_mesh_pose)}")
        
        if os.path.exists(ycb_mesh_pose):
            print(f"[DEBUG] YCB mesh_pose_dict count: {len(os.listdir(ycb_mesh_pose))}")
        if os.path.exists(acronym_mesh_pose):
            print(f"[DEBUG] ACRONYM mesh_pose_dict count: {len(os.listdir(acronym_mesh_pose))}")
        
        # Try to create VGNImplicit (this might fail but let's see the error)
        try:
            print("[DEBUG] Attempting to create VGNImplicit...")
            grasp_planner = VGNImplicit(
                temp_model_path,
                'targo_full_targ',
                best=True,
                qual_th=0.9,
                force_detection=True,
                out_th=0.5,
                select_top=False,
                visualize=False,
                cd_iou_measure=True,
            )
            print("[DEBUG] VGNImplicit created successfully")
        except Exception as e:
            print(f"[WARNING] VGNImplicit creation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test logging to wandb
        test_ycb_rate = 0.65
        test_acronym_rate = 0.72
        test_avg_rate = (test_ycb_rate + test_acronym_rate) / 2
        
        try:
            wandb.log({
                "val/grasp_success_rate": test_avg_rate,
                "val/ycb_success_rate": test_ycb_rate,
                "val/acronym_success_rate": test_acronym_rate,
                "debug/direct_test": 1.0
            }, step=0)
            print("[DEBUG] Successfully logged test metrics to wandb")
        except Exception as e:
            print(f"[ERROR] Failed to log to wandb: {e}")
        
        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print(f"[DEBUG] Cleaned up temporary model file")
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Finish wandb
        wandb.finish()
        print("✓ Test completed!")

if __name__ == "__main__":
    print("Testing direct validation function...")
    test_direct_validation()
    print("Done!") 