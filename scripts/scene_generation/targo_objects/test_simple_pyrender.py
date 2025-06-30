#!/usr/bin/env python3
"""
Simple test script for pyrender functionality.
"""

import os
import numpy as np
import trimesh
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_simple_pyrender():
    """Test basic pyrender functionality."""
    print("Testing basic pyrender functionality...")
    
    # Check if pyrender is available
    try:
        import pyrender
        from PIL import Image
        print("✓ pyrender package is available")
    except ImportError:
        print("✗ pyrender package is not available")
        print("Please install pyrender: pip install pyrender")
        return False
    
    # Create a simple cube mesh
    print("Creating test cube mesh...")
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    # Add colors to the faces
    colors = np.array([
        [255, 0, 0, 255],    # Red
        [0, 255, 0, 255],    # Green
        [0, 0, 255, 255],    # Blue
        [255, 255, 0, 255],  # Yellow
        [255, 0, 255, 255],  # Magenta
        [0, 255, 255, 255],  # Cyan
    ])
    cube.visual.face_colors = colors
    
    print(f"Cube mesh created: {len(cube.vertices)} vertices, {len(cube.faces)} faces")
    
    # Create pyrender scene
    print("Creating pyrender scene...")
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[255, 255, 255])
    
    # Add mesh to scene
    mesh_render = pyrender.Mesh.from_trimesh(cube, smooth=False)
    scene.add(mesh_render)
    
    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [2.0, 2.0, 2.0]  # Position camera
    scene.add(camera, pose=camera_pose)
    
    # Set up lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, -2, 2]
    scene.add(light, pose=light_pose)
    
    # Add ambient light
    ambient_light = pyrender.AmbientLight(color=[0.5, 0.5, 0.5], intensity=0.5)
    scene.add(ambient_light)
    
    # Render
    print("Rendering scene...")
    renderer = pyrender.OffscreenRenderer(800, 800)
    color, depth = renderer.render(scene)
    
    # Clean up
    renderer.delete()
    
    print(f"Rendered image shape: {color.shape}")
    print(f"Color range: {color.min()}-{color.max()}")
    
    # Save image
    output_dir = "demo"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_cube_rendered.png")
    
    image = Image.fromarray(color)
    image.save(output_path)
    print(f"✓ Test image saved to: {output_path}")
    
    # Check if file was created and has content
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✓ File created successfully, size: {file_size} bytes")
        if file_size > 0:
            print("✓ File has content (not empty)")
            return True
        else:
            print("✗ File is empty")
            return False
    else:
        print("✗ File was not created")
        return False

if __name__ == "__main__":
    success = test_simple_pyrender()
    if success:
        print("\n✓ Pyrender test completed successfully!")
    else:
        print("\n✗ Pyrender test failed!") 