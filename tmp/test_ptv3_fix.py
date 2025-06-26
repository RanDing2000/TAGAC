#!/usr/bin/env python3
"""
Test script for fixed PointTransformerV3 models
"""

import torch
import sys
import os

# Add project root to path
sys.path.append('.')

def test_ptv3_scene_model():
    """Test ptv3_scene model with dummy data"""
    print("Testing ptv3_scene model...")
    
    try:
        from src.vgn.networks import get_network
        
        # Create model
        net = get_network('ptv3_scene')
        print("✓ Model created successfully")
        
        # Create dummy data
        batch_size = 2
        num_points = 2048
        scene_pc = torch.randn(batch_size, num_points, 3)
        
        print(f"Input shape: {scene_pc.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = net.encoder_in(scene_pc)  # Test the encoder directly
            print(f"✓ Forward pass successful")
            print(f"Output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_targo_ptv3_model():
    """Test targo_ptv3 model with dummy data"""
    print("\nTesting targo_ptv3 model...")
    
    try:
        from src.vgn.networks import get_network
        
        # Create model
        net = get_network('targo_ptv3')
        print("✓ Model created successfully")
        
        # Create dummy data
        batch_size = 2
        num_points = 2048
        scene_pc = torch.randn(batch_size, num_points, 3)
        targ_pc = torch.randn(batch_size, num_points, 3)
        
        print(f"Scene input shape: {scene_pc.shape}")
        print(f"Target input shape: {targ_pc.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = net.encoder_in(scene_pc, targ_pc)  # Pass as separate arguments
            print(f"✓ Forward pass successful")
            print(f"Output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing fixed PointTransformerV3 models...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # Test ptv3_scene
    if test_ptv3_scene_model():
        success_count += 1
    
    # Test targo_ptv3
    if test_targo_ptv3_model():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("✓ All tests passed! Models are working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main() 