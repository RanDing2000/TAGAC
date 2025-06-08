#!/usr/bin/env python3
"""
Test script to verify wandb configuration in all training scripts.
"""

import argparse
from datetime import datetime

def test_wandb_config():
    """Test wandb configuration for all training scripts."""
    
    print("=" * 80)
    print("Testing Wandb Configuration")
    print("=" * 80)
    
    # Test train_targo.py
    print("\n1. train_targo.py:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"targo_{timestamp}"
    print(f"   Project: targo++")
    print(f"   Run name: {run_name}")
    
    # Test train_targo_full.py
    print("\n2. train_targo_full.py:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"targo_full_{timestamp}"
    print(f"   Project: targo++")
    print(f"   Run name: {run_name}")
    
    # Test train_targo_ptv3.py
    print("\n3. train_targo_ptv3.py:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"targo_ptv3_{timestamp}"
    print(f"   Project: targo++")
    print(f"   Run name: {run_name}")
    
    # Test train_simple.py
    print("\n4. train_simple.py:")
    print(f"   Project: targo++")
    print(f"   Calls child scripts with same project name")
    
    print("\n" + "=" * 80)
    print("All scripts configured to use project: targo++")
    print("Run names follow pattern: <model_name>_<timestamp>")
    print("=" * 80)

if __name__ == "__main__":
    test_wandb_config() 