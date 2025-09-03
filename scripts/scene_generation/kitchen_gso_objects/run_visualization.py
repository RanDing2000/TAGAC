#!/usr/bin/env python3
"""
Example script to run depth and segmentation visualization.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Example usage
    data_dir = Path("/home/ran.ding/projects/TARGO/messy_kitchen_scenes/gso_pile_scenes/visualizations/raw_data")
    output_dir = Path("/home/ran.ding/projects/TARGO/messy_kitchen_scenes/gso_pile_scenes/visualizations/results")
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please run the scene generation script first with --save-raw-data enabled")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run visualization script
    cmd = [
        sys.executable, "visualize_depth_seg.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--vis-depth",
        "--vis-segmentation"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Visualization completed successfully!")
        print("Output directory:", output_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

if __name__ == "__main__":
    main() 