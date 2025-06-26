#!/usr/bin/env python3
"""
Simple test to verify ptv3_scene configuration is correct.
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_function():
    """Test if get_model_ptv3_scene function is properly defined"""
    try:
        print("Testing get_model_ptv3_scene configuration function...")
        
        # Import the config module
        from src.vgn.ConvONets.conv_onet.config import get_model_ptv3_scene
        
        print("✓ Successfully imported get_model_ptv3_scene function")
        
        # Create a dummy config
        cfg = {
            'decoder': 'simple',
            'encoder': 'local_pool_pointnet',
            'c_dim': 64,
            'decoder_kwargs': {},
            'encoder_kwargs': {},
            'padding': 0.1,
        }
        
        print("✓ Created dummy configuration")
        
        # Test function signature (don't actually call it to avoid CUDA issues)
        import inspect
        sig = inspect.signature(get_model_ptv3_scene)
        print(f"✓ Function signature: {sig}")
        
        # Check if function is callable
        assert callable(get_model_ptv3_scene), "get_model_ptv3_scene should be callable"
        print("✓ Function is callable")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing config function: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_networks_import():
    """Test if ptv3_scene is properly registered in networks.py"""
    try:
        print("\nTesting networks.py registration...")
        
        # Check if ptv3_scene is in the networks dictionary
        from src.vgn.networks import NETWORKS
        
        if "ptv3_scene" in NETWORKS:
            print("✓ ptv3_scene is registered in NETWORKS dictionary")
            print(f"  ptv3_scene -> {NETWORKS['ptv3_scene']}")
            return True
        else:
            print("✗ ptv3_scene is NOT registered in NETWORKS dictionary")
            print(f"  Available networks: {list(NETWORKS.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing networks import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_construction_parameters():
    """Test if ConvolutionalOccupancyNetwork_Grid parameters are correct"""
    try:
        print("\nTesting model construction parameters...")
        
        # Import the models module
        from src.vgn.ConvONets.conv_onet.models import ConvolutionalOccupancyNetwork_Grid
        
        # Check constructor signature
        import inspect
        sig = inspect.signature(ConvolutionalOccupancyNetwork_Grid.__init__)
        print(f"✓ ConvolutionalOccupancyNetwork_Grid.__init__ signature: {sig}")
        
        # Verify expected parameters
        params = list(sig.parameters.keys())
        expected_params = ['self', 'decoders', 'encoders_in', 'encoder_aff', 'device', 'detach_tsdf', 'model_type']
        
        for param in expected_params:
            if param in params:
                print(f"✓ Parameter '{param}' found")
            else:
                print(f"✗ Parameter '{param}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model construction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 70)
    print("PTv3 Scene Configuration Test")
    print("=" * 70)
    
    # Test 1: Configuration function
    test1_passed = test_config_function()
    
    # Test 2: Networks registration
    test2_passed = test_networks_import()
    
    # Test 3: Model construction parameters
    test3_passed = test_model_construction_parameters()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"Config function test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Networks registration test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Model construction test: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✓ All configuration tests PASSED!")
        print("\nptv3_scene configuration is properly set up.")
        print("\nNext steps:")
        print("1. Load proper CUDA modules: module load compiler/gcc-8.3 && module load cuda/11.3.0")
        print("2. Activate environment: conda activate targo")
        print("3. Run training: python scripts/train_targo_ptv3.py --net ptv3_scene")
    else:
        print("\n✗ Some configuration tests FAILED. Please check the implementation.")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 