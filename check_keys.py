import numpy as np
from pathlib import Path

# Check a sample scene file to see what keys are available
dataset_path = Path('/storage/user/dira/nips_data_version6/combined/targo_dataset')
scenes_path = dataset_path / 'scenes'

# Find scene files
scene_files = list(scenes_path.glob('*.npz'))
print(f'Found {len(scene_files)} scene files')

if scene_files:
    # Separate single and clutter scenes
    single_scenes = [f for f in scene_files if '_s_' in f.name]
    clutter_scenes = [f for f in scene_files if '_c_' in f.name]
    
    print(f'Single scenes: {len(single_scenes)}')
    print(f'Clutter scenes: {len(clutter_scenes)}')
    
    # Check single scenes
    print('\n=== SINGLE SCENES (_s_) ===')
    for i, file in enumerate(single_scenes[:3]):
        try:
            data = np.load(file, allow_pickle=True)
            keys = list(data.keys())
            print(f'File {i+1}: {file.name}')
            print(f'  Keys: {keys}')
            
            if 'pc_scene_no_targ' not in data:
                print(f'  ❌ Missing pc_scene_no_targ')
            else:
                print(f'  ✓ Has pc_scene_no_targ')
            
            data.close()
        except Exception as e:
            print(f'Error loading {file.name}: {e}')
    
    # Check clutter scenes
    print('\n=== CLUTTER SCENES (_c_) ===')
    for i, file in enumerate(clutter_scenes[:3]):
        try:
            data = np.load(file, allow_pickle=True)
            keys = list(data.keys())
            print(f'File {i+1}: {file.name}')
            print(f'  Keys: {keys}')
            
            if 'pc_scene_no_targ' not in data:
                print(f'  ❌ Missing pc_scene_no_targ')
            else:
                print(f'  ✓ Has pc_scene_no_targ')
            
            data.close()
        except Exception as e:
            print(f'Error loading {file.name}: {e}')
else:
    print('No scene files found') 