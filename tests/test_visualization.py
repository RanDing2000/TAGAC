#!/usr/bin/env python3
"""
Test script for input data visualization functionality in train_targo_ptv3.py

This script creates synthetic point cloud data and tests the visualization function
to ensure it works correctly before running actual training.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import from train_targo_ptv3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the visualization function from the training script
try:
    from train_targo_ptv3 import visualize_input_data
    print("✓ Successfully imported visualize_input_data function")
except ImportError as e:
    print(f"✗ Failed to import visualization function: {e}")
    exit(1)

def create_synthetic_point_clouds():
    """Create synthetic scene and target point clouds for testing."""
    print("[TEST] Creating synthetic point cloud data...")
    
    # Create scene point cloud (scattered around center)
    scene_points = np.random.normal(0, 0.15, (1, 512, 3)).astype(np.float32)
    scene_points = np.clip(scene_points, -0.4, 0.4)  # Clip to valid range
    
    # Create target point cloud (more concentrated, shifted slightly)
    target_center = np.array([0.1, 0.1, 0.05])
    target_points = np.random.normal(0, 0.08, (1, 512, 3)).astype(np.float32) + target_center
    target_points = np.clip(target_points, -0.4, 0.4)  # Clip to valid range
    
    # Convert to tensors
    scene_tensor = torch.from_numpy(scene_points)
    target_tensor = torch.from_numpy(target_points)
    
    print(f"[TEST] Scene points shape: {scene_tensor.shape}")
    print(f"[TEST] Target points shape: {target_tensor.shape}")
    print(f"[TEST] Scene range: [{scene_points.min():.3f}, {scene_points.max():.3f}]")
    print(f"[TEST] Target range: [{target_points.min():.3f}, {target_points.max():.3f}]")
    
    return scene_tensor, target_tensor

def test_visualization_function():
    """Test the visualization function with synthetic data."""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION FUNCTION")
    print("="*60)
    
    # Create test data
    scene_points, target_points = create_synthetic_point_clouds()
    
    # Test parameters
    epoch = 1
    step = 100
    model_type = "targo_ptv3"
    vis_dir = Path("test_visualization_output")
    use_wandb = False  # Don't use wandb for testing
    
    print(f"[TEST] Testing visualization with:")
    print(f"  - Epoch: {epoch}")
    print(f"  - Step: {step}")
    print(f"  - Model type: {model_type}")
    print(f"  - Output directory: {vis_dir}")
    print(f"  - Wandb logging: {use_wandb}")
    
    try:
        # Call the visualization function
        print("\n[TEST] Calling visualize_input_data()...")
        visualize_input_data(
            scene_points, target_points,
            epoch, step, model_type,
            vis_dir, use_wandb=use_wandb
        )
        
        print("✓ Visualization function completed successfully!")
        
        # Check output files
        print("\n[TEST] Checking output files...")
        expected_files = [
            f"scene_epoch{epoch}_step{step}.ply",
            f"target_epoch{epoch}_step{step}.ply", 
            f"combined_epoch{epoch}_step{step}.ply",
            f"statistics_epoch{epoch}_step{step}.png",
            f"view_top_epoch{epoch}_step{step}.png",
            f"view_side1_epoch{epoch}_step{step}.png",
            f"view_side2_epoch{epoch}_step{step}.png",
            f"view_diagonal_epoch{epoch}_step{step}.png",
        ]
        
        missing_files = []
        for filename in expected_files:
            filepath = vis_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  ✓ {filename} ({size} bytes)")
            else:
                missing_files.append(filename)
                print(f"  ✗ {filename} (missing)")
        
        if missing_files:
            print(f"\n⚠️  Warning: {len(missing_files)} files are missing")
            return False
        else:
            print("\n✓ All expected files were created successfully!")
            return True
            
    except Exception as e:
        print(f"✗ Visualization function failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_shapes():
    """Test visualization with different input shapes."""
    print("\n" + "="*60)
    print("TESTING WITH DIFFERENT INPUT SHAPES")
    print("="*60)
    
    test_cases = [
        ("Standard 3D", (1, 512, 3)),
        ("Large point cloud", (1, 2048, 3)),
        ("Small point cloud", (1, 128, 3)),
        ("4D with labels", (1, 512, 4)),
    ]
    
    vis_dir = Path("test_visualization_shapes")
    
    for i, (test_name, shape) in enumerate(test_cases):
        print(f"\n[TEST {i+1}] {test_name} - Shape: {shape}")
        
        # Create test data with specified shape
        scene_points = np.random.normal(0, 0.15, shape).astype(np.float32)
        target_points = np.random.normal(0.1, 0.08, shape).astype(np.float32)
        
        # Clip to valid range
        scene_points = np.clip(scene_points, -0.4, 0.4)
        target_points = np.clip(target_points, -0.4, 0.4)
        
        scene_tensor = torch.from_numpy(scene_points)
        target_tensor = torch.from_numpy(target_points)
        
        try:
            visualize_input_data(
                scene_tensor, target_tensor,
                i+1, 1, "test", 
                vis_dir / f"test_{i+1}", 
                use_wandb=False
            )
            print(f"  ✓ {test_name} completed successfully")
        except Exception as e:
            print(f"  ✗ {test_name} failed: {e}")

def main():
    """Main test function."""
    print("Testing Input Data Visualization for train_targo_ptv3.py")
    print("="*60)
    
    # Test 1: Basic functionality
    success1 = test_visualization_function()
    
    # Test 2: Different input shapes
    test_with_different_shapes()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if success1:
        print("✓ Basic visualization test: PASSED")
        print("✓ All core functionality working correctly")
        print("\nThe visualization feature is ready for use in training!")
        print("\nTo use in training, run:")
        print("python scripts/train_targo_ptv3.py --enable_input_vis=True --vis_freq=100")
    else:
        print("✗ Basic visualization test: FAILED")
        print("Please check the error messages above and fix any issues.")
    
    print("\nTest output files saved in:")
    print("  - test_visualization_output/")
    print("  - test_visualization_shapes/")

if __name__ == "__main__":
    main() 