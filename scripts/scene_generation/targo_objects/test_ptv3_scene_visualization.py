import os
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_ptv3_scene_visualization():
    """
    Test script to verify ptv3_scene colored_scene_mesh visualization saving.
    """
    print("Testing ptv3_scene visualization saving...")
    
    # Check if demo directory exists and contains affordance visualization files
    demo_dir = Path("demo")
    if demo_dir.exists():
        print(f"Demo directory exists: {demo_dir}")
        
        # Look for affordance visualization files
        affordance_files = list(demo_dir.glob("*affordance_visual.obj"))
        if affordance_files:
            print(f"Found {len(affordance_files)} affordance visualization files:")
            for file in affordance_files:
                print(f"  - {file}")
                
                # Check file size
                file_size = file.stat().st_size
                print(f"    Size: {file_size} bytes")
                
                # Try to load the file with trimesh to verify it's valid
                try:
                    import trimesh
                    mesh = trimesh.load(str(file))
                    print(f"    Successfully loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                    
                    # Check if mesh has colors
                    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
                        colors = mesh.visual.face_colors
                        if colors is not None and len(colors) > 0:
                            print(f"    Mesh has colored faces: {len(colors)} face colors")
                            # Check color range
                            if len(colors.shape) > 1:
                                min_color = colors.min(axis=0)
                                max_color = colors.max(axis=0)
                                print(f"    Color range: {min_color} to {max_color}")
                        else:
                            print(f"    Mesh has no face colors")
                    else:
                        print(f"    Mesh has no visual attributes")
                        
                except Exception as e:
                    print(f"    Error loading mesh: {e}")
        else:
            print("No affordance visualization files found in demo directory")
    else:
        print("Demo directory does not exist")
    
    # Check if videos directory exists (for the formal saving location)
    videos_dir = Path("videos")
    if videos_dir.exists():
        print(f"Videos directory exists: {videos_dir}")
        
        # Look for affordance visualization files
        affordance_files = list(videos_dir.glob("*affordance_visual.obj"))
        if affordance_files:
            print(f"Found {len(affordance_files)} affordance visualization files in videos directory:")
            for file in affordance_files:
                print(f"  - {file}")
        else:
            print("No affordance visualization files found in videos directory")
    else:
        print("Videos directory does not exist")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_ptv3_scene_visualization() 