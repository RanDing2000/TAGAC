import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check_mask_targ_in_directory(scenes_dir, max_files=None):
    """
    Check which npz files don't have mask_targ data.
    
    Args:
        scenes_dir: Directory containing scene npz files
        max_files: Maximum number of files to check (for testing)
        
    Returns:
        no_mask_targ_files: List of scene_ids that don't have mask_targ
        total_files: Total number of files checked
    """
    scenes_dir = Path(scenes_dir)
    
    # Get all npz files
    npz_files = list(scenes_dir.glob("*.npz"))
    
    if max_files:
        npz_files = npz_files[:max_files]
    
    print(f"Found {len(npz_files)} npz files to check")
    
    no_mask_targ_files = []
    total_files = len(npz_files)
    
    # Check each npz file
    for npz_file in tqdm(npz_files, desc="Checking mask_targ existence"):
        scene_id = npz_file.stem
        
        try:
            # Load the npz file
            data = np.load(npz_file, allow_pickle=True)
            
            # Check if 'mask_targ' exists in the data
            if 'mask_targ' not in data:
                no_mask_targ_files.append(scene_id)
                print(f"Missing mask_targ: {scene_id}")
            
        except Exception as e:
            print(f"Error loading {scene_id}: {str(e)}")
            no_mask_targ_files.append(scene_id)
    
    return no_mask_targ_files, total_files

def main():
    parser = argparse.ArgumentParser(description="Test script to check mask_targ existence in npz files")
    parser.add_argument("--scenes_dir", type=str, 
                       default="data_scenes/targo_dataset/scenes",
                       help="Directory containing scene npz files")
    parser.add_argument("--max_files", type=int, default=10,
                       help="Maximum number of files to check (for testing)")
    
    args = parser.parse_args()
    
    print("Starting mask_targ existence check...")
    print(f"Scenes directory: {args.scenes_dir}")
    
    # Check mask_targ existence
    missing_files, total_files = check_mask_targ_in_directory(args.scenes_dir, args.max_files)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total files checked: {total_files}")
    print(f"Files missing mask_targ: {len(missing_files)}")
    print(f"Files with mask_targ: {total_files - len(missing_files)}")
    
    if missing_files:
        print(f"\n=== Scene IDs missing mask_targ ===")
        for scene_id in missing_files:
            print(f"  {scene_id}")
        
        # Save to tmp directory
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)
        
        no_mask_targ_file_path = tmp_dir / "no_mask_targ_file.txt"
        with open(no_mask_targ_file_path, 'w') as f:
            for scene_id in missing_files:
                f.write(f"{scene_id}\n")
        print(f"\nMissing scene IDs saved to: {no_mask_targ_file_path}")
    else:
        print(f"\n✅ All {total_files} files have mask_targ data!")

if __name__ == "__main__":
    main() 