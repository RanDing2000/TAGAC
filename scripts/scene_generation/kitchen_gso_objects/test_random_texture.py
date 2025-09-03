#!/usr/bin/env python3
"""
Test script for random texture functionality.
"""

import sys
from pathlib import Path
import random
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

def test_texture_discovery():
    """Test texture discovery functionality."""
    print("="*60)
    print("Testing Texture Discovery")
    print("="*60)
    
    # Check if texture directory exists
    if not TEXTURE_ROOT.exists():
        print(f"ERROR: Texture directory {TEXTURE_ROOT} does not exist!")
        print("Please run the download script first:")
        print("python data/download_cc_textures.py")
        return False
    
    # Get available textures
    textures = get_available_textures()
    
    if not textures:
        print("ERROR: No textures found!")
        return False
    
    print(f"Found {len(textures)} texture directories:")
    for i, texture in enumerate(textures[:10]):  # Show first 10
        print(f"  {i+1:2d}. {texture}")
    
    if len(textures) > 10:
        print(f"  ... and {len(textures) - 10} more")
    
    return True

def test_texture_loading():
    """Test texture loading functionality."""
    print("\n" + "="*60)
    print("Testing Texture Loading")
    print("="*60)
    
    textures = get_available_textures()
    if not textures:
        print("No textures available for testing")
        return False
    
    # Test loading a few random textures
    test_count = min(5, len(textures))
    test_textures = random.sample(textures, test_count)
    
    for i, texture_name in enumerate(test_textures):
        print(f"Testing texture {i+1}/{test_count}: {texture_name}")
        
        material = load_texture_material(texture_name)
        if material is not None:
            print(f"  ✓ Successfully loaded material")
            
            # Check what properties are available
            properties = []
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture:
                properties.append("diffuse")
            if hasattr(material, 'normalTexture') and material.normalTexture:
                properties.append("normal")
            if hasattr(material, 'roughnessFactor'):
                properties.append("roughness")
            if hasattr(material, 'metallicFactor'):
                properties.append("metallic")
            
            print(f"  Properties: {', '.join(properties)}")
        else:
            print(f"  ✗ Failed to load material")
    
    return True

def test_mesh_texture_application():
    """Test applying textures to meshes."""
    print("\n" + "="*60)
    print("Testing Mesh Texture Application")
    print("="*60)
    
    # Create a simple test mesh
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [0, 2, 3]]
    
    test_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"Created test mesh with {len(test_mesh.vertices)} vertices and {len(test_mesh.faces)} faces")
    
    # Get available textures
    textures = get_available_textures()
    if not textures:
        print("No textures available for testing")
        return False
    
    # Apply random texture
    print("Applying random texture...")
    textured_mesh = apply_random_texture_to_mesh(test_mesh, textures)
    
    if hasattr(textured_mesh.visual, 'material') and textured_mesh.visual.material:
        print("✓ Successfully applied texture to mesh")
        
        # Check material properties
        material = textured_mesh.visual.material
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture:
            print(f"  Diffuse texture: {material.baseColorTexture}")
        if hasattr(material, 'normalTexture') and material.normalTexture:
            print(f"  Normal texture: {material.normalTexture}")
    else:
        print("✗ Failed to apply texture to mesh")
        return False
    
    return True

def test_texture_cache():
    """Test texture caching functionality."""
    print("\n" + "="*60)
    print("Testing Texture Cache")
    print("="*60)
    
    textures = get_available_textures()
    if not textures:
        print("No textures available for testing")
        return False
    
    # Test texture
    test_texture = random.choice(textures)
    print(f"Testing cache with texture: {test_texture}")
    
    # Load texture twice
    print("Loading texture first time...")
    material1 = load_texture_material(test_texture)
    
    print("Loading texture second time (should use cache)...")
    material2 = load_texture_material(test_texture)
    
    if material1 is material2:
        print("✓ Cache working correctly (same object reference)")
    else:
        print("✗ Cache not working (different object references)")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Random Texture Functionality Test")
    print("="*60)
    
    tests = [
        ("Texture Discovery", test_texture_discovery),
        ("Texture Loading", test_texture_loading),
        ("Mesh Texture Application", test_mesh_texture_application),
        ("Texture Cache", test_texture_cache),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Random texture functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
