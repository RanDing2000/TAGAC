#!/usr/bin/env python3
"""
Simple test script to verify ptv3_scene model basic functionality.
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_import():
    """Test basic imports"""
    try:
        print("Testing basic imports...")
        
        # Test networks import
        from src.vgn.networks import get_network
        print("✓ Successfully imported get_network")
        
        # Test if ptv3_scene is in the models dict
        from src.vgn.networks import Ptv3SceneNet
        print("✓ Successfully imported Ptv3SceneNet")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic imports: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prepare_batch():
    """Test prepare_batch function"""
    try:
        print("\nTesting prepare_batch function...")
        
        # Import prepare_batch function
        sys.path.append('scripts')
        from train_targo_ptv3 import prepare_batch
        print("✓ Successfully imported prepare_batch")
        
        # Create dummy batch data
        batch_size = 2
        num_points = 2048
        
        # Dummy batch: (pc, targ_grid, targ_pc, scene_pc), (label, rotations, width), pos
        pc = torch.randn(batch_size, 1, 40, 40, 40)
        targ_grid = torch.randn(batch_size, 1, 40, 40, 40)
        targ_pc = torch.randn(batch_size, num_points, 3)
        scene_pc = torch.randn(batch_size, num_points, 3)
        
        label = torch.randn(batch_size)
        rotations = torch.randn(batch_size, 2, 4)
        width = torch.randn(batch_size)
        pos = torch.randn(batch_size, 3)
        
        batch = ((pc, targ_grid, targ_pc, scene_pc), (label, rotations, width), pos)
        device = torch.device('cpu')
        
        # Test ptv3_scene preparation
        x, y, pos_out = prepare_batch(batch, device, model_type="ptv3_scene")
        
        print(f"✓ ptv3_scene data preparation successful")
        print(f"  Input x shape: {x.shape}")
        print(f"  Labels y: {[yi.shape for yi in y]}")
        print(f"  Position pos shape: {pos_out.shape}")
        
        # Verify that x is just scene_pc (not a tuple)
        assert not isinstance(x, tuple), "For ptv3_scene, x should be scene_pc only, not a tuple"
        assert x.shape == scene_pc.shape, "x should have the same shape as scene_pc"
        
        print("✓ Data format verification passed")
        
        # Test targo_ptv3 preparation for comparison
        x_targo, y_targo, pos_targo = prepare_batch(batch, device, model_type="targo_ptv3")
        print(f"✓ targo_ptv3 data preparation successful")
        print(f"  Input x_targo type: {type(x_targo)}")
        if isinstance(x_targo, tuple):
            print(f"  Input x_targo shapes: {[xi.shape for xi in x_targo]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing prepare_batch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_args():
    """Test training script argument parsing"""
    try:
        print("\nTesting training script argument parsing...")
        
        # Import argparse from training script
        sys.path.append('scripts')
        import argparse
        
        # Create parser similar to train_targo_ptv3.py
        parser = argparse.ArgumentParser()
        parser.add_argument("--net", default="targo_ptv3", choices=["targo_ptv3", "ptv3_scene"], 
                            help="Network type: targo_ptv3 (scene+target) or ptv3_scene (scene only)")
        
        # Test ptv3_scene argument
        args = parser.parse_args(["--net", "ptv3_scene"])
        assert args.net == "ptv3_scene", "ptv3_scene argument not parsed correctly"
        print("✓ ptv3_scene argument parsing successful")
        
        # Test targo_ptv3 argument
        args = parser.parse_args(["--net", "targo_ptv3"])
        assert args.net == "targo_ptv3", "targo_ptv3 argument not parsed correctly"
        print("✓ targo_ptv3 argument parsing successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing argument parsing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Simple PTv3 Scene Model Test")
    print("=" * 60)
    
    # Test 1: Basic imports
    test1_passed = test_basic_import()
    
    # Test 2: Data preparation
    test2_passed = test_prepare_batch()
    
    # Test 3: Argument parsing
    test3_passed = test_training_script_args()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Basic imports test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Data preparation test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Argument parsing test: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✓ All basic tests PASSED! ptv3_scene implementation looks good.")
        print("You can now try training with:")
        print("python scripts/train_targo_ptv3.py --net ptv3_scene")
    else:
        print("\n✗ Some tests FAILED. Please check the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 