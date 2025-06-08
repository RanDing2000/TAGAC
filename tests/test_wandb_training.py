#!/usr/bin/env python3
"""
Test script to check wandb logging in training environment.
"""

import os
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

def test_wandb_training_environment():
    """Test wandb logging exactly like in training."""
    
    print("[DEBUG] Testing wandb in training environment...")
    
    # Initialize wandb exactly like in train_targo_full.py
    try:
        print("[DEBUG] Initializing wandb...")
        wandb.init(
            project="test-wandb-training",
            name=f"test-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": "targo_full",
                "test": True,
                "epochs": 3
            },
            tags=["test", "wandb-debugging"]
        )
        
        # Check if wandb initialization was successful
        print(f"[DEBUG] Wandb initialized successfully!")
        print(f"[DEBUG] Wandb run: {wandb.run}")
        print(f"[DEBUG] Wandb run id: {wandb.run.id}")
        print(f"[DEBUG] Wandb run name: {wandb.run.name}")
        print(f"[DEBUG] Wandb run url: {wandb.run.get_url()}")
        
        # Test logging a simple metric
        wandb.log({"debug/initialization_test": 1.0}, step=0)
        print("[DEBUG] Successfully logged initialization test metric to wandb")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize wandb: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test multiple epoch logging with validation metrics
    for epoch in range(1, 4):
        print(f"\n[DEBUG] ===== Testing Epoch {epoch} =====")
        
        # Test training metrics logging
        try:
            train_metrics = {
                "train/accuracy": 0.5 + epoch * 0.1,
                "train/loss_all": 2.0 - epoch * 0.3,
                "train/loss_qual": 1.0 - epoch * 0.15,
                "train/loss_rot": 0.8 - epoch * 0.1,
                "train/loss_width": 0.2 - epoch * 0.03
            }
            
            wandb.log(train_metrics, step=epoch)
            print(f"[DEBUG] Successfully logged training metrics for epoch {epoch}")
            
        except Exception as e:
            print(f"[ERROR] Failed to log training metrics for epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
        
        # Test validation metrics logging (simulate exactly like in training)
        try:
            # Simulate validation results
            ycb_success_rate = 0.2 + epoch * 0.1
            acronym_success_rate = 0.3 + epoch * 0.08
            avg_success_rate = (ycb_success_rate + acronym_success_rate) / 2
            
            print(f"[DEBUG] Simulated validation results:")
            print(f"[DEBUG]   YCB success rate: {ycb_success_rate:.4f}")
            print(f"[DEBUG]   ACRONYM success rate: {acronym_success_rate:.4f}")
            print(f"[DEBUG]   Average success rate: {avg_success_rate:.4f}")
            
            # Test wandb state check
            print(f"[DEBUG] Wandb run state: {wandb.run}")
            print(f"[DEBUG] Wandb run id: {wandb.run.id if wandb.run else 'None'}")
            print(f"[DEBUG] Wandb run name: {wandb.run.name if wandb.run else 'None'}")
            
            # Prepare metrics dictionary exactly like in training
            metrics_dict = {
                "val/grasp_success_rate": float(avg_success_rate),
                "val/ycb_success_rate": float(ycb_success_rate),
                "val/acronym_success_rate": float(acronym_success_rate),
                "debug/validation_source": 1.0,  # 1 = real validation
                "debug/epoch": int(epoch)
            }
            
            print(f"[DEBUG] Metrics to log: {metrics_dict}")
            
            # Log validation metrics
            wandb.log(metrics_dict, step=epoch)
            print(f"[DEBUG] Successfully logged validation metrics for epoch {epoch}")
            
            # Add some delay to simulate real training
            time.sleep(1)
            
        except Exception as e:
            print(f"[ERROR] Failed to log validation metrics for epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            
            # Try recovery like in training script
            try:
                print("[DEBUG] Attempting wandb recovery...")
                
                # Try logging with different approach
                simple_metrics = {
                    "val_simple/success_rate": float(avg_success_rate)
                }
                wandb.log(simple_metrics, step=epoch)
                print("[DEBUG] Successfully logged simplified validation metrics")
                
            except Exception as recovery_error:
                print(f"[ERROR] Wandb recovery also failed: {recovery_error}")
    
    # Test final wandb state
    try:
        print(f"\n[DEBUG] Final wandb run state: {wandb.run}")
        print(f"[DEBUG] Final wandb run url: {wandb.run.get_url()}")
        
        # Finish wandb
        wandb.finish()
        print("[DEBUG] Successfully finished wandb run")
        
    except Exception as e:
        print(f"[ERROR] Failed to finish wandb run: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Wandb training environment test completed!")
    return True

if __name__ == "__main__":
    print("Testing wandb in training environment...")
    success = test_wandb_training_environment()
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("Done!") 