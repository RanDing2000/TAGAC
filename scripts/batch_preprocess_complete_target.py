#!/usr/bin/env python3
"""
Batch preprocessing script to generate complete target meshes for all acronym and ycb datasets.

This script automatically processes all scene datasets in data_scenes/acronym and data_scenes/ycb
to generate complete target meshes and point clouds for training.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_preprocess_command(dataset_path, max_scenes=0):
    """
    Run the preprocessing command for a specific dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        max_scenes: Maximum number of scenes to process (0 = all)
    
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        sys.executable, "scripts/preprocess_complete_target_mesh.py",
        "--raw_root", str(dataset_path),
        "--output_root", str(dataset_path)
    ]
    
    if max_scenes > 0:
        cmd.extend(["--max_scenes", str(max_scenes)])
    
    print(f"\nProcessing dataset: {dataset_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Success!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {dataset_path}:")
        print(f"Return code: {e.returncode}")
        print(f"STDERR: {e.stderr}")
        print(f"STDOUT: {e.stdout}")
        return False

def main(args):
    """Main batch processing function."""
    base_path = Path(args.base_path)
    
    # Define all datasets to process
    acronym_datasets = [
        "acronym/acronym-slight-occlusion-1000",
        "acronym/acronym-middle-occlusion-1000", 
        "acronym/acronym-no-occlusion-1000"
    ]
    
    ycb_datasets = [
        "ycb/maniskill-ycb-v2-slight-occlusion-1000",
        "ycb/maniskill-ycb-v2-middle-occlusion-1000",
        "ycb/maniskill-ycb-v2-no-occlusion-1000"
    ]
    
    all_datasets = acronym_datasets + ycb_datasets
    
    print(f"Base path: {base_path}")
    print(f"Total datasets to process: {len(all_datasets)}")
    print("\nDatasets:")
    for dataset in all_datasets:
        print(f"  - {dataset}")
    
    # Process each dataset
    success_count = 0
    failed_datasets = []
    
    for dataset_name in all_datasets:
        dataset_path = base_path / dataset_name
        
        if not dataset_path.exists():
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            failed_datasets.append(dataset_name)
            continue
            
        # Check if required directories exist
        if not (dataset_path / "scenes").exists():
            print(f"Warning: scenes directory not found in {dataset_path}")
            failed_datasets.append(dataset_name)
            continue
            
        if not (dataset_path / "mesh_pose_dict").exists():
            print(f"Warning: mesh_pose_dict directory not found in {dataset_path}")
            failed_datasets.append(dataset_name)
            continue
        
        # Process the dataset
        if run_preprocess_command(dataset_path, args.max_scenes):
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total datasets: {len(all_datasets)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"Success rate: {success_count/len(all_datasets)*100:.1f}%")
    
    if failed_datasets:
        print(f"\nFailed datasets:")
        for dataset in failed_datasets:
            print(f"  - {dataset}")
    
    print(f"\nProcessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess complete target meshes for all datasets")
    parser.add_argument("--base_path", type=str, default="data_scenes",
                        help="Base path containing acronym and ycb dataset directories")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="Maximum number of scenes to process per dataset (0 = all)")
    
    args = parser.parse_args()
    main(args) 