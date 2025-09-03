#!/usr/bin/env python3
"""
Debug script to identify texture loading and application issues.
"""

import sys
from pathlib import Path
import trimesh
import numpy as np

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
    """Test texture discovery."""
    print("=== Testing Texture Discovery ===")
    textures = get_available_textures()
    print(f"Found {len(textures)} textures")
    if textures:
        print(f"First texture: {textures[0]}")
        return textures[0]
    return None

def test_texture_loading(texture_name):
    """Test texture loading."""
    print(f"\n=== Testing Texture Loading: {texture_name} ===")
    
    material = load_texture_material(texture_name)
    if material is None:
        print("Failed to load material")
        return None
    
    print(f"Material type: {type(material)}")
    print(f"Material attributes: {dir(material)}")
    
    # Check material properties
    for attr in ['baseColorTexture', 'baseColorFactor', 'normalTexture', 'roughnessFactor', 'metallicFactor']:
        if hasattr(material, attr):
            value = getattr(material, attr)
            print(f"  {attr}: {value}")
        else:
            print(f"  {attr}: Not found")
    
    return material

def test_mesh_creation():
    """Test mesh creation."""
    print("\n=== Testing Mesh Creation ===")
    
    # Create a simple cube mesh
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    print(f"Mesh visual type: {type(mesh.visual)}")
    print(f"Mesh visual attributes: {dir(mesh.visual)}")
    
    return mesh

def test_texture_application(mesh, material, texture_name):
    """Test texture application."""
    print(f"\n=== Testing Texture Application: {texture_name} ===")
    
    # Check mesh before texture application
    print("Before texture application:")
    print(f"  Mesh visual type: {type(mesh.visual)}")
    if hasattr(mesh.visual, 'material'):
        print(f"  Mesh has material: {mesh.visual.material is not None}")
    
    # Apply texture
    textured_mesh = apply_random_texture_to_mesh(mesh, [texture_name])
    
    # Check mesh after texture application
    print("After texture application:")
    print(f"  Mesh visual type: {type(textured_mesh.visual)}")
    if hasattr(textured_mesh.visual, 'material'):
        print(f"  Mesh has material: {textured_mesh.visual.material is not None}")
        if textured_mesh.visual.material:
            print(f"  Material type: {type(textured_mesh.visual.material)}")
            if hasattr(textured_mesh.visual.material, 'baseColorTexture'):
                print(f"  Material has baseColorTexture: {textured_mesh.visual.material.baseColorTexture}")
    
    return textured_mesh

def test_glb_export(mesh, texture_name):
    """Test GLB export."""
    print(f"\n=== Testing GLB Export: {texture_name} ===")
    
    try:
        # Export as GLB
        glb_path = f"/tmp/debug_texture_{texture_name}.glb"
        glb_data = mesh.export(file_type='glb')
        
        with open(glb_path, 'wb') as f:
            f.write(glb_data)
        
        print(f"Successfully exported GLB: {glb_path} (size: {len(glb_data)} bytes)")
        
        # Try to load back
        loaded_mesh = trimesh.load(glb_path)
        print(f"Successfully loaded GLB back")
        print(f"Loaded mesh visual type: {type(loaded_mesh.visual)}")
        
        return True
        
    except Exception as e:
        print(f"GLB export failed: {e}")
        return False

def main():
    """Run debug tests."""
    print("Texture Debug Test")
    print("="*60)
    
    # Step 1: Test texture discovery
    texture_name = test_texture_discovery()
    if not texture_name:
        print("No textures found, cannot continue")
        return False
    
    # Step 2: Test texture loading
    material = test_texture_loading(texture_name)
    if not material:
        print("Texture loading failed, cannot continue")
        return False
    
    # Step 3: Test mesh creation
    mesh = test_mesh_creation()
    
    # Step 4: Test texture application
    textured_mesh = test_texture_application(mesh, material, texture_name)
    
    # Step 5: Test GLB export
    success = test_glb_export(textured_mesh, texture_name)
    
    print("\n" + "="*60)
    print("Debug Summary")
    print("="*60)
    print(f"Texture discovery: {'✓ PASS' if texture_name else '✗ FAIL'}")
    print(f"Texture loading: {'✓ PASS' if material else '✗ FAIL'}")
    print(f"Mesh creation: {'✓ PASS' if mesh else '✗ FAIL'}")
    print(f"Texture application: {'✓ PASS' if textured_mesh else '✗ FAIL'}")
    print(f"GLB export: {'✓ PASS' if success else '✗ FAIL'}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
