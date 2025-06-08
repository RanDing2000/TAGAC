#!/usr/bin/env python3
"""
Generate occlusion level dictionary from existing scene files.
This script reads all .npz files in the scenes directory and extracts
the 'occ_targ' value to create the occlusion_level_dict.json file.

Usage:
    python scripts/generate_occlusion_level_from_scenes.py --input_dir data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000/scenes --output_file data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000/test_set/occlusion_level_dict.json
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def generate_occlusion_level_dict(input_dir, output_file):
    """
    Generate occlusion level dictionary from scene files.
    
    Args:
        input_dir (str): Directory containing .npz scene files
        output_file (str): Output path for the occlusion_level_dict.json file
    """
    print(f"Reading scene files from: {input_dir}")
    print(f"Output file: {output_file}")
    
    # Get all npz files from input directory
    scene_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    scene_files = sorted(scene_files)
    
    print(f"Found {len(scene_files)} scene files")
    
    if len(scene_files) == 0:
        print("No .npz files found in the input directory!")
        return
    
    # Dictionary to store occlusion levels
    occlusion_level_dict = {}
    
    # Process each scene file
    successful_files = 0
    failed_files = []
    
    for scene_file in tqdm(scene_files, desc="Processing scene files"):
        try:
            # Load the scene data
            scene_path = os.path.join(input_dir, scene_file)
            data = np.load(scene_path)
            
            # Check if 'occ_targ' key exists
            if 'occ_targ' not in data:
                print(f"Warning: 'occ_targ' not found in {scene_file}")
                failed_files.append(scene_file)
                continue
            
            # Extract scene name (remove .npz extension)
            scene_name = scene_file[:-4]
            
            # Get occlusion level value
            occ_level = float(data['occ_targ'])
            
            # Store in dictionary
            occlusion_level_dict[scene_name] = occ_level
            successful_files += 1
            
        except Exception as e:
            print(f"Error processing {scene_file}: {e}")
            failed_files.append(scene_file)
    
    print(f"\nProcessing completed:")
    print(f"  Successfully processed: {successful_files} files")
    print(f"  Failed: {len(failed_files)} files")
    
    if failed_files:
        print(f"  Failed files: {failed_files[:10]}...")  # Show first 10 failed files
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save the occlusion level dictionary
    with open(output_file, 'w') as f:
        json.dump(occlusion_level_dict, f, indent=2)
    
    print(f"\nOcclusion level dictionary saved to: {output_file}")
    print(f"Total entries: {len(occlusion_level_dict)}")
    
    # Show some statistics
    if occlusion_level_dict:
        occ_values = list(occlusion_level_dict.values())
        print(f"\nOcclusion level statistics:")
        print(f"  Min: {min(occ_values):.4f}")
        print(f"  Max: {max(occ_values):.4f}")
        print(f"  Mean: {np.mean(occ_values):.4f}")
        print(f"  Median: {np.median(occ_values):.4f}")
        
        # Show distribution in bins
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        bin_counts = np.histogram(occ_values, bins=bins)[0]
        bin_labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-1.0']
        
        print(f"\nOcclusion level distribution:")
        for label, count in zip(bin_labels, bin_counts):
            print(f"  {label}: {count} scenes")


def verify_occlusion_level_dict(dict_file, sample_size=5):
    """
    Verify the generated occlusion level dictionary by showing some samples.
    
    Args:
        dict_file (str): Path to the occlusion_level_dict.json file
        sample_size (int): Number of samples to show
    """
    print(f"\nVerifying occlusion level dictionary: {dict_file}")
    
    if not os.path.exists(dict_file):
        print(f"Error: File {dict_file} does not exist!")
        return
    
    with open(dict_file, 'r') as f:
        occ_dict = json.load(f)
    
    print(f"Total entries in dictionary: {len(occ_dict)}")
    
    # Show some samples
    items = list(occ_dict.items())
    sample_items = items[:sample_size]
    
    print(f"\nSample entries:")
    for scene_name, occ_level in sample_items:
        print(f"  {scene_name}: {occ_level}")


def main():
    parser = argparse.ArgumentParser(description='Generate occlusion level dictionary from scene files')
    parser.add_argument('--input_dir', type=str, 
                        default='data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000/scenes',
                        help='Directory containing .npz scene files')
    parser.add_argument('--output_file', type=str,
                        default='data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000/test_set/occlusion_level_dict.json',
                        help='Output path for occlusion_level_dict.json')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the generated file after creation')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_file = os.path.abspath(args.output_file)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    # Generate occlusion level dictionary
    generate_occlusion_level_dict(input_dir, output_file)
    
    # Verify if requested
    if args.verify:
        verify_occlusion_level_dict(output_file)


if __name__ == '__main__':
    main() 