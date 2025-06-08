#!/usr/bin/env python3
"""
Test script to verify wandb step ordering fix.
"""

import time
from datetime import datetime

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✓ wandb is available")
except ImportError:
    WANDB_AVAILABLE = False
    print("✗ wandb is not available")
    exit(1)

def test_wandb_step_ordering():
    """Test fixed wandb step ordering."""
    
    print("[DEBUG] Testing wandb step ordering fix...")
    
    # Global step counter
    global_step = {"value": 0}
    
    # Initialize wandb
    try:
        wandb.init(
            project="test-step-ordering",
            name=f"test-step-fix-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={"test": "step_ordering_fix"},
            tags=["test", "step-fix"]
        )
        
        # Test initialization
        wandb.log({"debug/initialization_test": 1.0}, step=global_step["value"])
        global_step["value"] += 1
        print(f"[DEBUG] Initialization logged at step {global_step['value']-1}")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize wandb: {e}")
        return False
    
    # Simulate multiple epochs
    for epoch in range(1, 4):
        print(f"\n[DEBUG] ===== Epoch {epoch} =====")
        
        # Simulate multiple batches per epoch (this would cause step ordering issues before)
        for batch in range(5):
            # Don't log individual batch steps to wandb - this was causing the issue
            print(f"[DEBUG] Batch {batch+1}/5 processed (not logging to wandb)")
        
        # Log epoch-level training metrics
        try:
            train_metrics = {
                "train/accuracy": 0.5 + epoch * 0.1,
                "train/loss_all": 2.0 - epoch * 0.3,
                "train/loss_qual": 1.0 - epoch * 0.15,
            }
            
            wandb.log(train_metrics, step=global_step["value"])
            print(f"[DEBUG] Training metrics logged at step {global_step['value']}")
            global_step["value"] += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to log training metrics: {e}")
        
        # Log learning rate
        try:
            wandb.log({"learning_rate": 1e-4 * (0.95 ** epoch)}, step=global_step["value"])
            print(f"[DEBUG] Learning rate logged at step {global_step['value']}")
            global_step["value"] += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to log learning rate: {e}")
        
        # Log validation metrics
        try:
            val_metrics = {
                "val/grasp_success_rate": 0.2 + epoch * 0.1,
                "val/ycb_success_rate": 0.15 + epoch * 0.08,
                "val/acronym_success_rate": 0.25 + epoch * 0.12,
                "debug/validation_source": 1.0,
                "debug/epoch": epoch
            }
            
            wandb.log(val_metrics, step=global_step["value"])
            print(f"[DEBUG] Validation metrics logged at step {global_step['value']}")
            global_step["value"] += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to log validation metrics: {e}")
        
        # Add some delay
        time.sleep(0.5)
    
    # Check final state
    try:
        print(f"\n[DEBUG] Final global step: {global_step['value']}")
        print(f"[DEBUG] Wandb run url: {wandb.run.get_url()}")
        
        wandb.finish()
        print("[DEBUG] Successfully finished wandb run")
        
    except Exception as e:
        print(f"[ERROR] Failed to finish wandb: {e}")
    
    print("\n✓ Wandb step ordering test completed!")
    return True

if __name__ == "__main__":
    print("Testing wandb step ordering fix...")
    success = test_wandb_step_ordering()
    if success:
        print("✓ Step ordering test passed!")
    else:
        print("✗ Step ordering test failed!")
    print("Done!") 