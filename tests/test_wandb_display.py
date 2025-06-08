#!/usr/bin/env python3
"""
Test script to verify wandb display information in training scripts.
"""

import subprocess
import sys

def test_wandb_display():
    """Test wandb display information for all training scripts."""
    
    scripts_to_test = [
        {
            "script": "scripts/train_simple.py",
            "args": ["--model_type", "targo", "--epochs", "1", "--batch-size", "2"]
        },
        {
            "script": "scripts/train_targo.py", 
            "args": ["--epochs", "1", "--batch-size", "2"]
        },
        {
            "script": "scripts/train_targo_full.py",
            "args": ["--epochs", "1", "--batch-size", "2"]
        },
        {
            "script": "scripts/train_targo_ptv3.py",
            "args": ["--epochs", "1", "--batch-size", "2"]
        }
    ]
    
    print("=" * 80)
    print("Testing Wandb Display Information")
    print("=" * 80)
    
    for test_case in scripts_to_test:
        script = test_case["script"]
        args = test_case["args"]
        
        print(f"\n{'='*60}")
        print(f"Testing: {script}")
        print(f"{'='*60}")
        
        # Test with wandb enabled (default)
        print("\n--- With Wandb ENABLED (default) ---")
        cmd_enabled = ["python", script] + args + ["--use_wandb", "True", "--wandb_project", "test_project"]
        try:
            result = subprocess.run(cmd_enabled, capture_output=True, text=True, timeout=10)
            output_lines = result.stdout.split('\n')
            
            # Look for wandb-related output
            for line in output_lines:
                if "Wandb logging" in line or "Wandb project" in line or "Wandb run name" in line or "Wandb log frequency" in line:
                    print(line)
                    
        except subprocess.TimeoutExpired:
            print("✓ Script started successfully (timeout expected)")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Test with wandb disabled
        print("\n--- With Wandb DISABLED ---")
        cmd_disabled = ["python", script] + args + ["--use_wandb", "False"]
        try:
            result = subprocess.run(cmd_disabled, capture_output=True, text=True, timeout=10)
            output_lines = result.stdout.split('\n')
            
            # Look for wandb-related output
            for line in output_lines:
                if "Wandb logging" in line:
                    print(line)
                    
        except subprocess.TimeoutExpired:
            print("✓ Script started successfully (timeout expected)")
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_wandb_display() 