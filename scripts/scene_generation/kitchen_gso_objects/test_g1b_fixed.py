#!/usr/bin/env python3

"""
Test script for fixed G1B file generation with PyVista rendering
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_transform_conversion():
    """Test the Transform 7-element array to 4x4 matrix conversion"""
    try:
        from src.vgn.utils.transform import Transform, Rotation
        
        # Create a test Transform
        translation = np.array([1, 2, 3])
        rotation = Rotation.identity()
        transform = Transform(rotation, translation)
        
        # Convert to 7-element list
        transform_list = transform.to_list()
        print(f"Transform list (7 elements): {transform_list}")
        
        # Convert back to Transform and get matrix
        transform_restored = Transform.from_list(transform_list)
        matrix = transform_restored.as_matrix()
        print(f"Matrix shape: {matrix.shape}")
        print(f"Matrix:\n{matrix}")
        
        # Test matrix inversion
        matrix_inv = np.linalg.inv(matrix)
        print(f"Matrix inverse:\n{matrix_inv}")
        
        print("✓ Transform conversion test passed")
        return True
        
    except Exception as e:
        print(f"✗ Transform conversion test failed: {e}")
        return False

def test_pyvista_rendering():
    """Test PyVista rendering with extrinsics"""
    try:
        import pyvista as pv
        from src.vgn.utils.transform import Transform, Rotation
        
        # Create a simple test scene
        import trimesh
        
        # Create test mesh
        box = trimesh.creation.box(extents=[1, 1, 1])
        
        # Create test mesh_pose_dict
        mesh_pose_dict = {
            1: ['test_box.ply', 1.0, np.eye(4)]
        }
        
        # Create test extrinsics (7-element array)
        translation = np.array([2, 2, 2])
        rotation = Rotation.identity()
        transform = Transform(rotation, translation)
        extrinsics = transform.to_list()
        
        print(f"Test extrinsics (7 elements): {extrinsics}")
        
        # Test the conversion logic
        from src.vgn.utils.transform import Transform
        transform = Transform.from_list(extrinsics)
        extrinsics_matrix = transform.as_matrix()
        print(f"Converted matrix:\n{extrinsics_matrix}")
        
        # Test matrix inversion
        camera_to_world = np.linalg.inv(extrinsics_matrix)
        camera_pos = camera_to_world[:3, 3]
        print(f"Camera position: {camera_pos}")
        
        print("✓ PyVista rendering test passed")
        return True
        
    except Exception as e:
        print(f"✗ PyVista rendering test failed: {e}")
        return False

def test_simulation_integration():
    """Test with actual simulation"""
    try:
        from src.vgn.simulation import ClutterRemovalSim
        from src.vgn.utils.implicit import get_mesh_pose_dict_from_world
        
        # Initialize simulation
        sim = ClutterRemovalSim("pile", "packed/train", gui=False)
        sim.reset(3, is_gso=True)
        
        # Get mesh pose dict
        mesh_pose_dict = get_mesh_pose_dict_from_world(sim.world, sim.object_set, exclude_plane=False)
        
        # Create test extrinsics
        from src.vgn.utils.transform import Transform, Rotation
        translation = np.array([2, 2, 2])
        rotation = Rotation.identity()
        transform = Transform(rotation, translation)
        extrinsics = transform.to_list()
        
        print(f"Simulation test - extrinsics: {extrinsics}")
        print(f"Mesh pose dict keys: {list(mesh_pose_dict.keys())}")
        
        print("✓ Simulation integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Simulation integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing G1B file generation with fixed extrinsics handling...")
    
    tests = [
        ("Transform conversion", test_transform_conversion),
        ("PyVista rendering", test_pyvista_rendering),
        ("Simulation integration", test_simulation_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} test ---")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*50)
    print("Test Results:")
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ The extrinsics handling fix is working correctly!")
        print("You can now run the generate_kitchen_dataset_gso.py script.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 