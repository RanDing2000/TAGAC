#!/usr/bin/env python3
"""
Simple test script for TARGO models with 1/10 scene sampling.

This script provides a unified interface for testing both:
- TARGO: Original TARGO model
- TARGO Full: Original TARGO model with complete target point clouds

Usage:
    python scripts/test_simple.py --model_type targo --model_path checkpoints/targonet.pt
    python scripts/test_simple.py --model_type targo_full --model_path checkpoints/targo_full.pt
"""

import argparse
import subprocess
import sys
import json
import random
from pathlib import Path
from datetime import datetime

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_test_scenes_subset(test_root, subset_ratio=0.1):
    """Get a subset of test scenes for quick evaluation."""
    test_root = Path(test_root)
    
    # Find all scene directories
    scene_dirs = []
    test_set_dir = test_root / "test_set"
    if test_set_dir.exists():
        scene_dirs = [d for d in test_set_dir.iterdir() if d.is_dir()]
    
    if not scene_dirs:
        print(f"Warning: No test scenes found in {test_set_dir}")
        return []
    
    # Sample subset
    num_scenes = max(1, int(len(scene_dirs) * subset_ratio))
    random.seed(42)  # For reproducible results
    selected_scenes = random.sample(scene_dirs, num_scenes)
    
    print(f"Selected {num_scenes}/{len(scene_dirs)} scenes for testing")
    return [scene.name for scene in selected_scenes]

def test_ycb(args):
    """Test on YCB dataset."""
    print("=" * 60)
    print(f"Testing {args.model_type.upper()} (Original TARGO) on YCB Dataset")
    print("=" * 60)
    
    # Determine model type for inference script
    if args.model_type == "targo":
        inference_type = "targo"
    elif args.model_type == "targo_full":
        inference_type = "targo_full_targ"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    cmd = [
        "python", "scripts/inference_ycb.py",
        "--type", inference_type,
        "--model", str(args.model_path),
        "--occlusion-level", args.occlusion_level,
        "--result_root", str(args.result_root),
        "--qual-th", str(args.qual_th),
        "--out_th", str(args.out_th),
        "--best",
        "--force",
        "--sideview"
    ]
    
    if args.vis:
        cmd.append("--vis")
    
    if args.video_recording:
        cmd.extend(["--video-recording", "True"])
    
    if args.target_file:
        cmd.extend(["--target-file", str(args.target_file)])
    
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd)

def test_acronym(args):
    """Test on ACRONYM dataset."""
    print("=" * 60)
    print(f"Testing {args.model_type.upper()} (Original TARGO) on ACRONYM Dataset")
    print("=" * 60)
    
    # Determine model type for inference script
    if args.model_type == "targo":
        inference_type = "targo"
    elif args.model_type == "targo_full":
        inference_type = "targo_full_targ"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    cmd = [
        "python", "scripts/inference_acronym.py",
        "--type", inference_type,
        "--model", str(args.model_path),
        "--occlusion-level", args.occlusion_level,
        "--result_root", str(args.result_root),
        "--qual-th", str(args.qual_th),
        "--out_th", str(args.out_th),
        "--best",
        "--force",
        "--sideview"
    ]
    
    if args.vis:
        cmd.append("--vis")
    
    if args.video_recording:
        cmd.extend(["--video-recording", "True"])
    
    if args.target_file:
        cmd.extend(["--target-file", str(args.target_file)])
    
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd)

def create_subset_test_config(args):
    """Create a test configuration with 1/10 scenes."""
    if args.dataset_type == "ycb":
        test_root_map = {
            "medium": "data_scenes/ycb/maniskill-ycb-v2-middle-occlusion-1000",
            "slight": "data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000", 
            "no": "data_scenes/ycb/maniskill-ycb-v2-no-occlusion-1000"
        }
    else:  # acronym
        test_root_map = {
            "medium": "data_scenes/acronym/acronym-middle-occlusion-1000",
            "slight": "data_scenes/acronym/acronym-slight-occlusion-1000",
            "no": "data_scenes/acronym/acronym-no-occlusion-1000"
        }
    
    test_root = test_root_map.get(args.occlusion_level)
    if not test_root:
        print(f"Unknown occlusion level: {args.occlusion_level}")
        return None
    
    # Get subset of scenes
    selected_scenes = get_test_scenes_subset(test_root, args.subset_ratio)
    
    if not selected_scenes:
        print("No test scenes found!")
        return None
    
    # Create subset configuration
    subset_config = {
        "test_root": test_root,
        "selected_scenes": selected_scenes,
        "subset_ratio": args.subset_ratio,
        "total_scenes": len(selected_scenes),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save configuration
    config_path = args.result_root / f"test_subset_config_{args.dataset_type}_{args.occlusion_level}.json"
    with open(config_path, 'w') as f:
        json.dump(subset_config, f, indent=2)
    
    print(f"Test subset configuration saved to: {config_path}")
    return subset_config

def main():
    parser = argparse.ArgumentParser(description="Simple test script for original TARGO models")
    
    # Model selection
    parser.add_argument("--model_type", choices=["targo", "targo_full"], required=True,
                        help="Model type: targo (original) or targo_full (original with complete target)")
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to model checkpoint")
    
    # Dataset selection
    parser.add_argument("--dataset_type", choices=["ycb", "acronym"], default="ycb",
                        help="Dataset type: ycb or acronym")
    parser.add_argument("--occlusion_level", choices=["no", "slight", "medium"], default="medium",
                        help="Occlusion level: no, slight, or medium")
    
    # Test parameters
    parser.add_argument("--subset_ratio", type=float, default=0.1,
                        help="Ratio of scenes to test (default: 0.1 for 1/10)")
    parser.add_argument("--qual_th", type=float, default=0.9,
                        help="Quality threshold for valid grasps")
    parser.add_argument("--out_th", type=float, default=0.5,
                        help="Output threshold for valid grasps")
    
    # Output settings
    parser.add_argument("--result_root", type=Path,
                        default="test_results_simple",
                        help="Root directory for test results")
    parser.add_argument("--vis", action="store_true", default=False,
                        help="Enable visualization")
    parser.add_argument("--video_recording", type=str2bool, default=False,
                        help="Enable video recording")
    parser.add_argument("--target_file", type=Path, default=None,
                        help="Path to target file for video recording")
    
    args = parser.parse_args()
    
    # Create result directory
    args.result_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Simple TARGO Test Script")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Occlusion level: {args.occlusion_level}")
    print(f"Subset ratio: {args.subset_ratio} ({args.subset_ratio*100:.1f}%)")
    print(f"Result root: {args.result_root}")
    print("=" * 60)
    
    # Verify model file exists
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create test subset configuration
    subset_config = create_subset_test_config(args)
    if not subset_config:
        print("Failed to create test configuration")
        sys.exit(1)
    
    # Run test based on dataset type
    if args.dataset_type == "ycb":
        result = test_ycb(args)
    elif args.dataset_type == "acronym":
        result = test_acronym(args)
    else:
        print(f"Unknown dataset type: {args.dataset_type}")
        sys.exit(1)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Testing completed successfully!")
        print(f"Results saved to: {args.result_root}")
        print(f"Tested {subset_config['total_scenes']} scenes")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Testing failed!")
        print(f"Return code: {result.returncode}")
        print("=" * 60)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main() 