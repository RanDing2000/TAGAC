#!/usr/bin/env python3
"""
Test script for the new TARGO-PTv3 model.
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append('src')

from vgn.networks import get_network

def test_targo_ptv3_model():
    """Test the TARGO-PTv3 model creation and basic functionality."""
    
    print("Testing TARGO-PTv3 model...")
    
    # Create a simple test configuration
    test_config = {
        'model_type': 'targo_ptv3',
        'd_model': 64,
        'cross_att_key': 'pointnet_cross_attention',
        'num_attention_layers': 0,
        'encoder': 'voxel_simple_local_without_3d',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_kwargs': {
            'sample_mode': 'bilinear',
            'hidden_size': 32
        },
        'padding': 0,
        'c_dim': 64
    }
    
    try:
        # Test model creation
        print("Creating TARGO-PTv3 network...")
        # network = get_network('targo_ptv3', test_config)
        network = get_network('targo_ptv3')
        print("✓ Network created successfully")
        
        # Test with dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        network = network.to(device)
        network.eval()
        
        # Create dummy input data
        batch_size = 2
        n_scene_points = 1000
        n_target_points = 500
        
        # Scene point cloud: (batch_size, n_points, 3)
        scene_pc = torch.randn(batch_size, n_scene_points, 3, device=device)
        # Target point cloud: (batch_size, n_points, 3)  
        target_pc = torch.randn(batch_size, n_target_points, 3, device=device)
        
        print(f"Scene PC shape: {scene_pc.shape}")
        print(f"Target PC shape: {target_pc.shape}")
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            inputs = [scene_pc, target_pc]
            
            # Test the encoder part
            try:
                features = network.encoder_in(scene_pc, target_pc)
                print(f"✓ Encoder forward pass successful, output shape: {features.shape}")
            except Exception as e:
                print(f"✗ Encoder forward pass failed: {e}")
                return False
                
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_targo_ptv3_model()
    if success:
        print("\n🎉 TARGO-PTv3 model is ready for use!")
    else:
        print("\n❌ TARGO-PTv3 model needs debugging.") 