#!/usr/bin/env python3
"""
Simple test script to debug validation wandb logging issues.
"""

import torch
import tempfile
import os
from pathlib import Path

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✓ wandb is available")
except ImportError:
    WANDB_AVAILABLE = False
    print("✗ wandb is not available")
    exit(1)

def test_validation_function():
    """Test simplified validation function."""
    
    print("[DEBUG] Testing validation function...")
    
    # Initialize wandb
    wandb.init(
        project="test-validation",
        name="test-validation-run",
        config={"test": True}
    )
    
    # Test basic validation data logging
    for epoch in range(3):
        print(f"[DEBUG] Testing epoch {epoch}")
        
        # Simulate validation results
        ycb_success_rate = 0.6 + epoch * 0.05
        acronym_success_rate = 0.7 + epoch * 0.03
        avg_success_rate = (ycb_success_rate + acronym_success_rate) / 2
        
        print(f"[DEBUG] YCB success rate: {ycb_success_rate:.4f}")
        print(f"[DEBUG] ACRONYM success rate: {acronym_success_rate:.4f}")
        print(f"[DEBUG] Average success rate: {avg_success_rate:.4f}")
        
        # Log to wandb
        try:
            wandb.log({
                "val/grasp_success_rate": avg_success_rate,
                "val/ycb_success_rate": ycb_success_rate,
                "val/acronym_success_rate": acronym_success_rate,
                "debug/test_counter": epoch
            }, step=epoch)
            print(f"[DEBUG] Successfully logged validation metrics to wandb for epoch {epoch}")
        except Exception as e:
            print(f"[ERROR] Failed to log to wandb: {e}")
    
    # Check if validation datasets exist
    ycb_test_root = 'data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000'
    acronym_test_root = 'data_scenes/acronym/acronym-slight-occlusion-1000'
    
    print(f"[DEBUG] Checking YCB test root: {ycb_test_root}")
    print(f"[DEBUG] YCB path exists: {os.path.exists(ycb_test_root)}")
    print(f"[DEBUG] Checking ACRONYM test root: {acronym_test_root}")
    print(f"[DEBUG] ACRONYM path exists: {os.path.exists(acronym_test_root)}")
    
    if os.path.exists(ycb_test_root):
        print(f"[DEBUG] YCB test scenes: {len(os.listdir(ycb_test_root + '/test_set/scenes'))}")
    
    if os.path.exists(acronym_test_root):
        print(f"[DEBUG] ACRONYM test scenes: {len(os.listdir(acronym_test_root + '/test_set/scenes'))}")
    
    # Finish wandb
    wandb.finish()
    print("✓ Test completed successfully!")

if __name__ == "__main__":
    print("Testing validation wandb logging...")
    test_validation_function()
    print("Done!") 