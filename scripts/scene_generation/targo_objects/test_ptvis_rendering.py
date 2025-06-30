import os
import numpy as np
import sys
import trimesh
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def render_colored_scene_mesh_with_pyrender(colored_scene_mesh, output_path="demo/ptv3_scene_affordance_visual.png", 
                                          width=800, height=600, camera_distance=0.5):
    """
    Render colored scene mesh using pyrender package.
    
    Args:
        colored_scene_mesh: trimesh object with colored faces
        output_path: path to save the rendered image
        width: image width
        height: image height
        camera_distance: distance from camera to scene center
    """
    if not PYRENDER_AVAILABLE:
        print("pyrender not available, skipping rendering")
        return False
    
    try:
        # Create pyrender scene
        scene = pyrender.Scene()
        
        # Add the colored mesh to the scene
        scene.add(pyrender.Mesh.from_trimesh(colored_scene_mesh, smooth=False))
        
        # Set up camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [camera_distance, camera_distance, camera_distance]
        scene.add(camera, pose=camera_pose)
        
        # Set up lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, :3] = np.eye(3) * 1.0
        light_pose[:3, 3] = [0, -1, -1]
        scene.add(light, pose=light_pose)
        
        # Render the scene
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(scene)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the rendered image
        image = Image.fromarray(color)
        image.save(output_path)
        print(f"Rendered image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error rendering with pyrender: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_ptvis_rendering():
    """
    Test script to verify ptvis rendering functionality.
    """
    print("Testing ptvis rendering functionality...")
    
    # Check if ptvis is available
    try:
        import ptvis
        print("✓ ptvis package is available")
    except ImportError:
        print("✗ ptvis package is not available")
        print("Please install ptvis: pip install ptvis")
        return False
    
    # Check if demo directory exists and contains affordance visualization files
    demo_dir = Path("demo")
    if not demo_dir.exists():
        print("Demo directory does not exist, creating test mesh...")
        
        # Create a simple test mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ])
        
        # Create colored mesh
        test_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        test_mesh.visual.face_colors = np.array([
            [255, 0, 0, 255],  # Red
            [0, 255, 0, 255],  # Green
            [0, 0, 255, 255],  # Blue
            [255, 255, 0, 255]  # Yellow
        ])
        
        # Save test mesh
        demo_dir.mkdir(exist_ok=True)
        test_mesh.export("demo/test_mesh.obj")
        print("Created test mesh: demo/test_mesh.obj")
        
        # Test rendering with ptvis
        # from src.vgn.detection_ptv3_implicit import render_colored_scene_mesh_with_ptvis
        
        success = render_colored_scene_mesh_with_ptvis(
            test_mesh,
            output_path="demo/test_mesh_rendered.png",
            width=800,
            height=600,
            camera_distance=2.0
        )
        
        if success:
            print("✓ Successfully rendered test mesh with ptvis")
            print("Rendered image saved to: demo/test_mesh_rendered.png")
        else:
            print("✗ Failed to render test mesh with ptvis")
            return False
    
    else:
        print(f"Demo directory exists: {demo_dir}")
        
        # Look for affordance visualization files
        affordance_files = list(demo_dir.glob("*affordance_visual.obj"))
        if affordance_files:
            print(f"Found {len(affordance_files)} affordance visualization files:")
            for file in affordance_files:
                print(f"  - {file}")
                
                # Try to load and render the mesh
                try:
                    mesh = trimesh.load(str(file))
                    print(f"    Successfully loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                    
                    # Test rendering with ptvis
                    # from src.vgn.detection_ptv3_implicit import render_colored_scene_mesh_with_ptvis
                    
                    # Generate output path
                    output_path = str(file).replace('.obj', '_rendered.png')
                    
                    success = render_colored_scene_mesh_with_ptvis(
                        mesh,
                        output_path=output_path,
                        width=800,
                        height=600,
                        camera_distance=0.5
                    )
                    
                    if success:
                        print(f"    ✓ Successfully rendered: {output_path}")
                    else:
                        print(f"    ✗ Failed to render: {output_path}")
                        
                except Exception as e:
                    print(f"    Error processing {file}: {e}")
        else:
            print("No affordance visualization files found in demo directory")
    
    # Check for rendered images
    rendered_files = list(demo_dir.glob("*_rendered.png"))
    if rendered_files:
        print(f"\nFound {len(rendered_files)} rendered images:")
        for file in rendered_files:
            file_size = file.stat().st_size
            print(f"  - {file} ({file_size} bytes)")
    else:
        print("\nNo rendered images found")
    
    print("\nTest completed!")
    return True

if __name__ == "__main__":
    test_ptvis_rendering() 