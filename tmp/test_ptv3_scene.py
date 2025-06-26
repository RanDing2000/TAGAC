#!/usr/bin/env python3
"""
Test script to verify ptv3_scene model functionality.
"""

import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ptv3_scene_import():
    """Test if ptv3_scene can be imported and created"""
    try:
        from src.vgn.networks import get_network
        print("✓ Successfully imported get_network")
        
        # Test creating ptv3_scene network
        net = get_network("ptv3_scene")
        print("✓ Successfully created ptv3_scene network")
        
        # Check if it's a valid PyTorch model
        assert isinstance(net, torch.nn.Module), "Network should be a PyTorch module"
        print("✓ ptv3_scene is a valid PyTorch module")
        
        return True, net
        
    except Exception as e:
        print(f"✗ Error testing ptv3_scene import: {e}")
        return False, None

def test_ptv3_scene_forward():
    """Test forward pass of ptv3_scene model"""
    try:
        success, net = test_ptv3_scene_import()
        if not success:
            return False
            
        # Create dummy input data (scene point cloud only)
        batch_size = 2
        num_points = 2048
        
        # Scene point cloud [B, N, 3]
        scene_pc = torch.randn(batch_size, num_points, 3)
        
        # Grasp position [B, 1, 3]
        pos = torch.randn(batch_size, 1, 3)
        
        print(f"✓ Created dummy input: scene_pc {scene_pc.shape}, pos {pos.shape}")
        
        # Test forward pass
        net.eval()
        with torch.no_grad():
            output = net(scene_pc, pos)
            
        print(f"✓ Forward pass successful, output shape: {[o.shape if hasattr(o, 'shape') else type(o) for o in output]}")
        
        # Check output format (should be qual, rot, width)
        if isinstance(output, (list, tuple)) and len(output) == 3:
            qual_out, rot_out, width_out = output
            print(f"✓ Output format correct: qual {qual_out.shape}, rot {rot_out.shape}, width {width_out.shape}")
        else:
            print(f"✗ Unexpected output format: {type(output)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing ptv3_scene forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preparation():
    """Test data preparation for ptv3_scene"""
    try:
        # Import prepare_batch function
        sys.path.append('scripts')
        from train_targo_ptv3 import prepare_batch
        
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
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("PTv3 Scene Model Test")
    print("=" * 60)
    
    # Test 1: Import and creation
    test1_passed = test_ptv3_scene_import()[0]
    
    # Test 2: Forward pass
    test2_passed = test_ptv3_scene_forward()
    
    # Test 3: Data preparation
    test3_passed = test_data_preparation()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Import and creation test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Forward pass test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Data preparation test: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✓ All tests PASSED! ptv3_scene model is ready for training.")
    else:
        print("\n✗ Some tests FAILED. Please check the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 