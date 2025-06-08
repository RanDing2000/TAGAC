#!/usr/bin/env python3
"""
Debug validation error by testing the validation data loader.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import random

def debug_validation_error():
    print("Testing validation data loader...")
    
    # Test the validation loader creation function
    try:
        from scripts.train_targo import create_validation_loader_from_slight_occlusion
        
        # Same parameters as in training
        batch_size = 128
        data_contain = "pc and targ_grid"
        decouple = False
        use_complete_targ = False
        model_type = "targo"
        input_points = 'tsdf_points'
        shape_completion = True
        vis_data = False
        logdir = None
        kwargs = {"num_workers": 1, "pin_memory": True}
        
        print("Creating validation loader...")
        val_loader = create_validation_loader_from_slight_occlusion(
            batch_size, data_contain, decouple, use_complete_targ, model_type, 
            input_points, shape_completion, vis_data, logdir, kwargs
        )
        
        print(f"Validation loader created with {len(val_loader.dataset)} samples")
        
        # Try to load first batch
        print("Attempting to load validation batches...")
        for i, batch in enumerate(val_loader):
            print(f"Successfully loaded validation batch {i}")
            if i >= 2:  # Load a few batches
                break
        
        print("Validation data loading successful!")
        
    except Exception as e:
        print(f"Error during validation data loading: {e}")
        import traceback
        traceback.print_exc()
        
        # Check the validation data files directly
        print("\nChecking validation data files...")
        
        ycb_slight_path = Path('data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000')
        acronym_slight_path = Path('data_scenes/acronym/acronym-slight-occlusion-1000')
        
        val_scene_paths = []
        
        # Check YCB files
        if ycb_slight_path.exists():
            ycb_scenes_path = ycb_slight_path / 'scenes'
            if ycb_scenes_path.exists():
                ycb_scene_files = list(ycb_scenes_path.glob('*_c_*.npz'))
                print(f"Found {len(ycb_scene_files)} YCB validation scenes")
                if len(ycb_scene_files) >= 100:
                    ycb_selected = random.sample(ycb_scene_files, 100)
                else:
                    ycb_selected = ycb_scene_files
                val_scene_paths.extend(ycb_selected)
                
                # Check a few YCB files
                for i, scene_path in enumerate(ycb_selected[:5]):
                    try:
                        scene_data = np.load(scene_path, allow_pickle=True)
                        keys = list(scene_data.keys())
                        print(f"YCB scene {scene_path.name}: {keys}")
                        if 'grid_scene' not in keys:
                            print(f"  *** MISSING grid_scene in {scene_path.name} ***")
                        scene_data.close()
                    except Exception as file_error:
                        print(f"  Error reading YCB file {scene_path.name}: {file_error}")
        
        # Check ACRONYM files
        if acronym_slight_path.exists():
            acronym_scenes_path = acronym_slight_path / 'scenes'
            if acronym_scenes_path.exists():
                acronym_scene_files = list(acronym_scenes_path.glob('*_c_*.npz'))
                print(f"Found {len(acronym_scene_files)} ACRONYM validation scenes")
                if len(acronym_scene_files) >= 100:
                    acronym_selected = random.sample(acronym_scene_files, 100)
                else:
                    acronym_selected = acronym_scene_files
                val_scene_paths.extend(acronym_selected)
                
                # Check a few ACRONYM files
                for i, scene_path in enumerate(acronym_selected[:5]):
                    try:
                        scene_data = np.load(scene_path, allow_pickle=True)
                        keys = list(scene_data.keys())
                        print(f"ACRONYM scene {scene_path.name}: {keys}")
                        if 'grid_scene' not in keys:
                            print(f"  *** MISSING grid_scene in {scene_path.name} ***")
                        scene_data.close()
                    except Exception as file_error:
                        print(f"  Error reading ACRONYM file {scene_path.name}: {file_error}")

if __name__ == "__main__":
    debug_validation_error() 