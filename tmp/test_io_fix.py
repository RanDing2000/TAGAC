import sys
sys.path.append('src')

from vgn.io import read_scene_no_targ_pc
from pathlib import Path
import numpy as np

# Test the fixed function
dataset_path = Path('/storage/user/dira/nips_data_version6/combined/targo_dataset')

# Test with a single scene
single_scenes = list(dataset_path.glob('scenes/*_s_*.npz'))
if single_scenes:
    scene_file = single_scenes[0]
    scene_id = scene_file.stem
    print(f"Testing single scene: {scene_id}")
    try:
        scene_pc = read_scene_no_targ_pc(dataset_path, scene_id)
        print(f"✓ Successfully loaded single scene point cloud: {scene_pc.shape}")
    except Exception as e:
        print(f"❌ Failed to load single scene: {e}")

# Test with a clutter scene
clutter_scenes = list(dataset_path.glob('scenes/*_c_*.npz'))
if clutter_scenes:
    scene_file = clutter_scenes[0]
    scene_id = scene_file.stem
    print(f"\nTesting clutter scene: {scene_id}")
    try:
        scene_pc = read_scene_no_targ_pc(dataset_path, scene_id)
        print(f"✓ Successfully loaded clutter scene point cloud: {scene_pc.shape}")
    except Exception as e:
        print(f"❌ Failed to load clutter scene: {e}")

print("\nTest completed!") 