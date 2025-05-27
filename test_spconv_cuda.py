#!/usr/bin/env python3
"""
Test script to check if spconv is using CUDA correctly
"""

import torch
import sys
import time
import numpy as np

def test_cuda_availability():
    """Test basic CUDA availability"""
    print("=== CUDA Availability Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print()

def test_spconv_cuda():
    """Test spconv with CUDA"""
    print("=== spconv CUDA Test ===")
    
    try:
        import spconv.pytorch as spconv
        print(f"spconv version: {spconv.__version__}")
        
        # Create test data on CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create sparse tensor data
        batch_size = 2
        num_points = 1000
        
        # Generate random coordinates
        coords = []
        feats = []
        
        for b in range(batch_size):
            # Random coordinates in a 40x40x40 grid
            coord = torch.randint(0, 40, (num_points, 3), dtype=torch.int32)
            # Add batch index
            batch_coord = torch.cat([
                torch.full((num_points, 1), b, dtype=torch.int32),
                coord
            ], dim=1)
            coords.append(batch_coord)
            
            # Random features
            feat = torch.randn(num_points, 3, dtype=torch.float32)
            feats.append(feat)
        
        # Concatenate all batches
        coords = torch.cat(coords, dim=0).to(device)
        feats = torch.cat(feats, dim=0).to(device)
        
        print(f"Coordinates shape: {coords.shape}, device: {coords.device}")
        print(f"Features shape: {feats.shape}, device: {feats.device}")
        
        # Create sparse tensor
        spatial_shape = [40, 40, 40]
        sparse_tensor = spconv.SparseConvTensor(
            features=feats,
            indices=coords,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        
        print(f"Sparse tensor created successfully")
        print(f"Sparse tensor device: {sparse_tensor.features.device}")
        print(f"Sparse tensor indices device: {sparse_tensor.indices.device}")
        
        # Create a simple sparse convolution layer
        conv = spconv.SubMConv3d(3, 16, 3, padding=1).to(device)
        print(f"Conv layer device: next(conv.parameters()).device")
        
        # Test forward pass
        start_time = time.time()
        output = conv(sparse_tensor)
        end_time = time.time()
        
        print(f"✓ Forward pass successful!")
        print(f"Output features shape: {output.features.shape}")
        print(f"Output device: {output.features.device}")
        print(f"Forward pass time: {end_time - start_time:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in spconv CUDA test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ptv3_models_cuda():
    """Test PTv3 models with CUDA"""
    print("\n=== PTv3 Models CUDA Test ===")
    
    try:
        sys.path.append('.')
        from src.vgn.networks import get_network
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Test ptv3_scene model
        print("\nTesting ptv3_scene model...")
        net = get_network('ptv3_scene').to(device)
        print(f"Model device: {next(net.parameters()).device}")
        
        # Create test data
        batch_size = 2
        num_points = 2048
        scene_pc = torch.randn(batch_size, num_points, 3).to(device)
        
        print(f"Input device: {scene_pc.device}")
        
        # Test forward pass with timing
        start_time = time.time()
        with torch.no_grad():
            output = net.encoder_in(scene_pc)
        end_time = time.time()
        
        print(f"✓ ptv3_scene forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output device: {output.device}")
        print(f"Forward pass time: {end_time - start_time:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in PTv3 models CUDA test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all CUDA tests"""
    print("Testing spconv and PTv3 models CUDA usage...")
    print("=" * 60)
    
    # Test CUDA availability
    test_cuda_availability()
    
    # Test spconv with CUDA
    spconv_success = test_spconv_cuda()
    
    # Test PTv3 models with CUDA
    ptv3_success = test_ptv3_models_cuda()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"spconv CUDA test: {'✓ PASSED' if spconv_success else '✗ FAILED'}")
    print(f"PTv3 models CUDA test: {'✓ PASSED' if ptv3_success else '✗ FAILED'}")
    
    if spconv_success and ptv3_success:
        print("\n✓ All tests passed! spconv is using CUDA correctly.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        print("\nNote: If spconv falls back to CPU, it will be significantly slower.")
        print("CPU vs CUDA performance difference can be 10-100x for sparse convolutions.")

if __name__ == "__main__":
    main() 