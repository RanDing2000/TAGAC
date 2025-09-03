#!/usr/bin/env python3
"""
Simple test script to verify texture loading functionality.
"""

import sys
from pathlib import Path
import trimesh

# Add TARGO to path
sys.path.append('/home/ran.ding/projects/TARGO')

# Import texture functions from the main script
from generate_kitchen_dataset_gso_scene_random_texture import (
    get_available_textures, 
    load_texture_material, 
    apply_random_texture_to_mesh,
    TEXTURE_ROOT
)

def test_single_texture():
    """Test loading a single texture."""
    print("Testing single texture loading...")
    
    # Get available textures
    textures = get_available_textures()
    if not textures:
        print("No textures available!")
        return False
    
    # Test with first texture
    test_texture = textures[0]
    print(f"Testing texture: {test_texture}")
    
    # Load material
    material = load_texture_material(test_texture)
    if material is None:
        print("Failed to load material!")
        return False
    
    print(f"Successfully loaded material for {test_texture}")
    
    # Check material properties
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture:
        print(f"  Diffuse texture: {material.baseColorTexture}")
    else:
        print("  No diffuse texture found")
    
    if hasattr(material, 'normalTexture') and material.normalTexture:
        print(f"  Normal texture: {material.normalTexture}")
    else:
        print("  No normal texture found")
    
    return True

def test_mesh_texture_application():
    """Test applying texture to a mesh."""
    print("\nTesting mesh texture application...")
    
    # Create a simple test mesh
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [0, 2, 3]]
    
    test_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"Created test mesh with {len(test_mesh.vertices)} vertices and {len(test_mesh.faces)} faces")
    
    # Get available textures
    textures = get_available_textures()
    if not textures:
        print("No textures available!")
        return False
    
    # Apply texture
    textured_mesh = apply_random_texture_to_mesh(test_mesh, textures)
    
    # Check if texture was applied
    if hasattr(textured_mesh.visual, 'material') and textured_mesh.visual.material:
        print("✓ Successfully applied texture to mesh")
        
        material = textured_mesh.visual.material
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture:
            print(f"  Diffuse texture: {material.baseColorTexture}")
        
        # Try to export as GLB to test
        try:
            glb_data = textured_mesh.export(file_type='glb')
            print(f"✓ Successfully exported GLB with texture (size: {len(glb_data)} bytes)")
            return True
        except Exception as e:
            print(f"✗ Failed to export GLB: {e}")
            return False
    else:
        print("✗ Failed to apply texture to mesh")
        return False

def main():
    """Run tests."""
    print("Texture Loading Test")
    print("="*50)
    
    # Test 1: Single texture loading
    test1_result = test_single_texture()
    
    # Test 2: Mesh texture application
    test2_result = test_mesh_texture_application()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Single texture loading: {'✓ PASS' if test1_result else '✗ FAIL'}")
    print(f"Mesh texture application: {'✓ PASS' if test2_result else '✗ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! Texture loading is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
