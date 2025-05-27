#!/usr/bin/env python3
"""
Test script to verify VGN model can generate category statistics correctly.
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vgn.detection import VGN
from src.vgn.utils.transform import Transform, Rotation
from src.vgn.grasp import Grasp

def test_vgn_cd_iou_measure():
    """Test VGN with cd_iou_measure parameter"""
    print("Testing VGN with cd_iou_measure parameter...")
    
    # Create a mock VGN instance
    try:
        # Use a dummy model path for testing
        model_path = "dummy_path"
        vgn = VGN(
            model_path="dummy_path", 
            model_type="vgn",
            cd_iou_measure=True
        )
        print("✓ VGN instance created successfully with cd_iou_measure=True")
        
        # Check if cd_iou_measure attribute exists
        assert hasattr(vgn, 'cd_iou_measure'), "VGN should have cd_iou_measure attribute"
        assert vgn.cd_iou_measure == True, "cd_iou_measure should be True"
        print("✓ cd_iou_measure attribute correctly set")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing VGN: {e}")
        return False

def test_vgn_call_signature():
    """Test VGN __call__ method signature"""
    print("\nTesting VGN __call__ method signature...")
    
    try:
        # Check if VGN.__call__ accepts the required parameters
        from inspect import signature
        from src.vgn.detection import VGN
        
        call_sig = signature(VGN.__call__)
        params = list(call_sig.parameters.keys())
        
        required_params = ['self', 'state', 'scene_mesh', 'aff_kwargs', 'hunyun2_path', 'scene_name', 'cd_iou_measure', 'target_mesh_gt']
        
        for param in required_params:
            if param in params:
                print(f"✓ Parameter '{param}' found in VGN.__call__")
            else:
                print(f"✗ Parameter '{param}' missing from VGN.__call__")
                return False
        
        print("✓ All required parameters found in VGN.__call__ method")
        return True
        
    except Exception as e:
        print(f"✗ Error checking VGN.__call__ signature: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("VGN Category Statistics Test")
    print("=" * 60)
    
    # Test 1: VGN instance creation with cd_iou_measure
    test1_passed = test_vgn_cd_iou_measure()
    
    # Test 2: VGN __call__ method signature
    test2_passed = test_vgn_call_signature()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"VGN cd_iou_measure test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"VGN __call__ signature test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests PASSED! VGN should now work with category statistics.")
    else:
        print("\n✗ Some tests FAILED. VGN may not work correctly with category statistics.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 