import os
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check_mask_targ_existence(scenes_dir, max_files=None):
    """
    Check if each npz file contains 'mask_targ' data.
    
    Args:
        scenes_dir: Directory containing scene npz files
        max_files: Maximum number of files to check (for testing)
        
    Returns:
        missing_mask_targ_files: List of scene_ids that don't have mask_targ
        total_files: Total number of files checked
    """
    scenes_dir = Path(scenes_dir)
    
    # Get all npz files
    npz_files = list(scenes_dir.glob("*.npz"))
    
    if max_files:
        npz_files = npz_files[:max_files]
    
    print(f"Found {len(npz_files)} npz files to check")
    
    missing_mask_targ_files = []
    total_files = len(npz_files)
    
    # Check each npz file
    for npz_file in tqdm(npz_files, desc="Checking mask_targ existence"):
        scene_id = npz_file.stem
        
        try:
            # Load the npz file
            data = np.load(npz_file, allow_pickle=True)
            
            # Check if 'mask_targ' exists in the data
            if 'mask_targ' not in data:
                missing_mask_targ_files.append(scene_id)
                print(f"Missing mask_targ: {scene_id}")
            
        except Exception as e:
            print(f"Error loading {scene_id}: {str(e)}")
            missing_mask_targ_files.append(scene_id)
    
    return missing_mask_targ_files, total_files

def main():
    parser = argparse.ArgumentParser(description="Test script to check mask_targ existence in npz files")
    parser.add_argument("--scenes_dir", type=str, 
                       default="/usr/stud/dira/GraspInClutter/targo/data_scenes/targo_dataset/scenes",
                       help="Directory containing scene npz files")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to check (for testing)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save missing mask_targ scene_ids (optional)")
    
    args = parser.parse_args()
    
    print("Starting mask_targ existence check...")
    print(f"Scenes directory: {args.scenes_dir}")
    
    # Check mask_targ existence
    missing_files, total_files = check_mask_targ_existence(args.scenes_dir, args.max_files)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total files checked: {total_files}")
    print(f"Files missing mask_targ: {len(missing_files)}")
    print(f"Files with mask_targ: {total_files - len(missing_files)}")
    
    if missing_files:
        print(f"\n=== Scene IDs missing mask_targ ===")
        for scene_id in missing_files:
            print(f"  {scene_id}")
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(missing_files, f, indent=2)
            print(f"\nMissing scene IDs saved to: {args.output_file}")
    else:
        print(f"\n✅ All {total_files} files have mask_targ data!")

if __name__ == "__main__":
    main() 