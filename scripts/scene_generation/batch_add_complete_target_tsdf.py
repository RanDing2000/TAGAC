#!/usr/bin/env python3
"""
Batch script to add complete_target_tsdf to both ACRONYM and YCB dataset scenes.
This script can process both datasets sequentially or individually.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_script(script_path, dataset_path, max_scenes=None, dry_run=False):
    """
    Run a processing script with given parameters.
    
    Args:
        script_path: Path to the processing script
        dataset_path: Path to the dataset
        max_scenes: Maximum number of scenes to process
        dry_run: Whether to run in dry-run mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = ['python', script_path, '--dataset_path', dataset_path]
    
    if max_scenes:
        cmd.extend(['--max_scenes', str(max_scenes)])
    
    if dry_run:
        cmd.append('--dry_run')
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch add complete_target_tsdf to ACRONYM and YCB datasets')
    parser.add_argument('--datasets', nargs='+', choices=['acronym', 'ycb', 'all'], 
                       default=['all'], help='Which datasets to process')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='Maximum number of scenes to process per dataset (default: all)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Analyze files without modifying them')
    parser.add_argument('--acronym_path', type=str,
                       default='/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym-slight-occlusion-1000',
                       help='Path to ACRONYM dataset')
    parser.add_argument('--ycb_path', type=str,
                       default='/usr/stud/dira/GraspInClutter/targo/data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000',
                       help='Path to YCB dataset')
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    datasets_to_process = []
    if 'all' in args.datasets:
        datasets_to_process = ['acronym', 'ycb']
    else:
        datasets_to_process = args.datasets
    
    print("Batch Complete Target TSDF Generator")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets to process: {', '.join(datasets_to_process)}")
    print(f"Max scenes per dataset: {args.max_scenes if args.max_scenes else 'All'}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dataset configurations
    dataset_configs = {
        'acronym': {
            'script': os.path.join(script_dir, 'acronym_objects', 'add_complete_target_tsdf.py'),
            'path': args.acronym_path,
            'name': 'ACRONYM'
        },
        'ycb': {
            'script': os.path.join(script_dir, 'ycb_objects', 'add_complete_target_tsdf.py'),
            'path': args.ycb_path,
            'name': 'YCB'
        }
    }
    
    # Process each dataset
    results = {}
    
    for dataset in datasets_to_process:
        config = dataset_configs[dataset]
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {config['name']} DATASET")
        print(f"{'='*60}")
        print(f"Dataset path: {config['path']}")
        print(f"Script path: {config['script']}")
        print()
        
        # Check if dataset path exists
        if not os.path.exists(config['path']):
            print(f"Error: Dataset path does not exist: {config['path']}")
            results[dataset] = False
            continue
        
        # Check if script exists
        if not os.path.exists(config['script']):
            print(f"Error: Script does not exist: {config['script']}")
            results[dataset] = False
            continue
        
        # Run the processing script
        success = run_script(
            config['script'], 
            config['path'], 
            args.max_scenes, 
            args.dry_run
        )
        
        results[dataset] = success
        
        if success:
            print(f"\n✓ {config['name']} dataset processing completed successfully")
        else:
            print(f"\n✗ {config['name']} dataset processing failed")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for dataset in datasets_to_process:
        config = dataset_configs[dataset]
        status = "✓ SUCCESS" if results[dataset] else "✗ FAILED"
        print(f"{config['name']}: {status}")
    
    # Overall success
    all_success = all(results.values())
    print(f"\nOverall result: {'✓ ALL SUCCESSFUL' if all_success else '✗ SOME FAILED'}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 