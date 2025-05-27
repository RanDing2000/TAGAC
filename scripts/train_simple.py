#!/usr/bin/env python3
"""
Simple training script for TARGO models.

This script provides a unified interface for training both:
- TARGO: Original TARGO model with shape completion using AdaPoinTr
- TARGO Full: Original TARGO model using preprocessed complete target point clouds without shape completion

Usage:
    python scripts/train_simple.py --model_type targo --epochs 50
    python scripts/train_simple.py --model_type targo_full --epochs 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_targo(args):
    """Train original TARGO model with shape completion."""
    print("=" * 60)
    print("Training Original TARGO with Shape Completion")
    print("=" * 60)
    
    cmd = [
        "python", "scripts/train_targo_ptv3.py",
        "--net", "targo_ptv3",
        "--dataset", str(args.dataset),
        "--dataset_raw", str(args.dataset_raw),
        "--logdir", str(args.logdir / "targo"),
        "--data_contain", "pc and targ_grid",
        "--use_complete_targ", "False",
        "--shape_completion", "True",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--val-split", str(args.val_split),
        "--vis_data", "False",
        "--description", "targo_with_shape_completion",
        "--lr-schedule-interval", str(args.lr_schedule_interval),
        "--gamma", str(args.gamma)
    ]
    
    if args.augment:
        cmd.append("--augment")
    
    if args.ablation_dataset:
        cmd.extend(["--ablation_dataset", args.ablation_dataset])
    
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd)

def train_targo_full(args):
    """Train original TARGO Full model with complete target point clouds."""
    print("=" * 60)
    print("Training Original TARGO Full with Complete Target Point Clouds")
    print("=" * 60)
    
    cmd = [
        "python", "scripts/train_targo_full.py",
        "--net", "targo",
        "--dataset", str(args.dataset),
        "--dataset_raw", str(args.dataset_raw),
        "--logdir", str(args.logdir / "targo_full"),
        "--data_contain", "pc and targ_grid",
        "--use_complete_targ", "True",
        "--shape_completion", "False",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--val-split", str(args.val_split),
        "--vis_data", "False",
        "--description", "targo_full_complete_target",
        "--lr-schedule-interval", str(args.lr_schedule_interval),
        "--gamma", str(args.gamma)
    ]
    
    if args.augment:
        cmd.append("--augment")
    
    if args.ablation_dataset:
        cmd.extend(["--ablation_dataset", args.ablation_dataset])
    
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Simple training script for original TARGO models")
    
    # Model selection
    parser.add_argument("--model_type", choices=["targo", "targo_full"], required=True,
                        help="Model type: targo (original with shape completion) or targo_full (original with complete target)")
    
    # Dataset paths
    parser.add_argument("--dataset", type=Path, 
                        default='/storage/user/dira/nips_data_version6/combined/targo_dataset',
                        help="Path to processed dataset")
    parser.add_argument("--dataset_raw", type=Path,
                        default='/storage/user/dira/nips_data_version6/combined/targo_dataset',
                        help="Path to raw dataset")
    parser.add_argument("--logdir", type=Path,
                        default="/usr/stud/dira/GraspInClutter/grasping/train_logs_simple",
                        help="Base log directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--lr-schedule-interval", type=int, default=10, 
                        help="Learning rate schedule interval")
    parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate decay factor")
    
    # Dataset options
    parser.add_argument("--ablation_dataset", type=str, default="", 
                        help="Ablation dataset option: 1_10 | 1_100 | only_cluttered")
    
    args = parser.parse_args()
    
    # Create log directory
    args.logdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Simple TARGO Training Script")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Run training based on model type
    if args.model_type == "targo":
        result = train_targo(args)
    elif args.model_type == "targo_full":
        result = train_targo_full(args)
    else:
        print(f"Unknown model type: {args.model_type}")
        sys.exit(1)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Check logs at: {args.logdir}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training failed!")
        print(f"Return code: {result.returncode}")
        print("=" * 60)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main() 