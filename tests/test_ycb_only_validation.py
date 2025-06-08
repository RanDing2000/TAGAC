#!/usr/bin/env python3
"""
Test script to verify YCB-only validation works correctly.
"""

import torch
import tempfile
import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = "/usr/stud/dira/GraspInClutter/targo"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_ycb_only_validation():
    """Test YCB-only validation function."""
    
    print("[DEBUG] Testing YCB-only validation...")
    
    # Import the validation function from the training script
    from scripts.train_targo_full import perform_validation_grasp_evaluation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logdir = Path("/tmp/test_ycb_validation")
    logdir.mkdir(exist_ok=True)
    epoch = 1
    
    # Create a dummy model state dict (mimicking net.state_dict())
    dummy_model_data = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
    }
    
    # Create a dummy network object with state_dict method
    class DummyNet:
        def state_dict(self):
            return dummy_model_data
    
    net = DummyNet()
    
    try:
        # Test the validation function
        print(f"[DEBUG] Running validation test...")
        ycb_rate, acronym_rate, avg_rate = perform_validation_grasp_evaluation(net, device, logdir, epoch)
        
        print(f"[TEST RESULT] YCB success rate: {ycb_rate:.4f}")
        print(f"[TEST RESULT] ACRONYM success rate: {acronym_rate:.4f} (should be 0.0)")
        print(f"[TEST RESULT] Average success rate: {avg_rate:.4f} (should equal YCB rate)")
        
        # Verify results
        if acronym_rate == 0.0:
            print("✓ ACRONYM validation correctly disabled")
        else:
            print("✗ ACRONYM validation not properly disabled")
            
        if avg_rate == ycb_rate:
            print("✓ Average rate correctly equals YCB rate")
        else:
            print("✗ Average rate calculation incorrect")
            
        if ycb_rate >= 0.0:
            print("✓ YCB validation returned valid result")
        else:
            print("✗ YCB validation returned invalid result")
            
        print("\n[DEBUG] YCB-only validation test completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing YCB-only validation functionality...")
    success = test_ycb_only_validation()
    if success:
        print("✓ YCB-only validation test passed!")
    else:
        print("✗ YCB-only validation test failed!")
    print("Done!") 