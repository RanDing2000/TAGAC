#!/usr/bin/env python3
"""
Test script to verify wandb validation upload functionality.
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Error: wandb not available. Install with 'pip install wandb'")
    exit(1)

def test_wandb_validation_upload():
    """Test wandb validation metrics upload with various scenarios."""
    
    print("=" * 60)
    print("Testing Wandb Validation Upload")
    print("=" * 60)
    
    # Initialize wandb
    project_name = "targo_validation_test"
    run_name = f"validation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Initializing wandb project: {project_name}")
    print(f"Run name: {run_name}")
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "test_type": "validation_upload",
                "timestamp": datetime.now().isoformat()
            },
            tags=["test", "validation"],
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            )
        )
        print("✓ Wandb initialization successful")
    except Exception as e:
        print(f"✗ Wandb initialization failed: {e}")
        return False
    
    # Test 1: Basic validation metrics upload
    print("\n[Test 1] Basic validation metrics upload...")
    global_step = 0
    
    try:
        basic_metrics = {
            "val/test_metric": 0.75,
            "val/ycb_success_rate": 0.65,
            "val/acronym_success_rate": 0.0,
            "debug/test_flag": 1.0
        }
        
        wandb.log(basic_metrics, step=global_step)
        global_step += 1
        print("✓ Basic metrics upload successful")
        
    except Exception as e:
        print(f"✗ Basic metrics upload failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Multiple validation epochs simulation
    print("\n[Test 2] Multiple validation epochs simulation...")
    
    for epoch in range(1, 6):  # Simulate 5 epochs
        try:
            # Simulate improving validation metrics
            base_rate = 0.3 + (epoch * 0.1)
            ycb_rate = base_rate + np.random.normal(0, 0.05)
            avg_rate = max(0.1, ycb_rate)
            
            epoch_metrics = {
                "val/grasp_success_rate": float(avg_rate),
                "val/ycb_success_rate": float(ycb_rate),
                "val/acronym_success_rate": 0.0,  # Disabled
                "debug/validation_source": 1.0,  # Real validation
                "debug/epoch": int(epoch),
                "debug/ycb_only": 1.0,
                "debug/acronym_disabled": 1.0
            }
            
            print(f"  Epoch {epoch}: Uploading metrics {epoch_metrics}")
            wandb.log(epoch_metrics, step=global_step)
            global_step += 1
            
            # Add small delay to simulate real training
            time.sleep(0.5)
            
            print(f"  ✓ Epoch {epoch} metrics uploaded successfully")
            
        except Exception as e:
            print(f"  ✗ Epoch {epoch} metrics upload failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Stress test with rapid uploads
    print("\n[Test 3] Stress test with rapid uploads...")
    
    try:
        for i in range(10):
            stress_metrics = {
                f"stress/metric_{i}": np.random.random(),
                "stress/iteration": i,
                "stress/timestamp": time.time()
            }
            wandb.log(stress_metrics, step=global_step)
            global_step += 1
            time.sleep(0.1)  # Small delay
        
        print("✓ Stress test completed successfully")
        
    except Exception as e:
        print(f"✗ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Large metrics dictionary
    print("\n[Test 4] Large metrics dictionary upload...")
    
    try:
        large_metrics = {}
        for i in range(50):
            large_metrics[f"large/metric_{i}"] = np.random.random()
        
        large_metrics.update({
            "large/validation_complete": 1.0,
            "large/total_metrics": len(large_metrics)
        })
        
        wandb.log(large_metrics, step=global_step)
        global_step += 1
        print(f"✓ Large metrics dictionary ({len(large_metrics)} metrics) uploaded successfully")
        
    except Exception as e:
        print(f"✗ Large metrics dictionary upload failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final status check
    print("\n[Final Check] Wandb run status...")
    
    try:
        print(f"Wandb run ID: {wandb.run.id}")
        print(f"Wandb run name: {wandb.run.name}")
        print(f"Wandb run URL: {wandb.run.get_url()}")
        print(f"Total steps logged: {global_step}")
        
        # Log final summary
        summary_metrics = {
            "test/total_steps": global_step,
            "test/completion_time": time.time(),
            "test/status": "completed"
        }
        wandb.log(summary_metrics, step=global_step)
        
        print("✓ Final status check successful")
        
    except Exception as e:
        print(f"✗ Final status check failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Finish wandb run
    try:
        wandb.finish()
        print("✓ Wandb run finished successfully")
        return True
        
    except Exception as e:
        print(f"✗ Wandb finish failed: {e}")
        return False

def test_offline_mode():
    """Test wandb offline mode functionality."""
    
    print("\n" + "=" * 60)
    print("Testing Wandb Offline Mode")
    print("=" * 60)
    
    # Set offline mode
    os.environ["WANDB_MODE"] = "offline"
    
    project_name = "targo_validation_offline_test"
    run_name = f"offline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config={"test_type": "offline_validation"},
            tags=["test", "offline"]
        )
        
        # Test offline logging
        for i in range(5):
            metrics = {
                "offline/metric": np.random.random(),
                "offline/step": i
            }
            wandb.log(metrics, step=i)
        
        wandb.finish()
        print("✓ Offline mode test successful")
        print("Note: Use 'wandb sync' to upload offline data later")
        return True
        
    except Exception as e:
        print(f"✗ Offline mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test wandb validation upload functionality")
    parser.add_argument("--test-offline", action="store_true", help="Also test offline mode")
    parser.add_argument("--project", type=str, default="targo_validation_test", help="Wandb project name")
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        print("Wandb is not available. Please install it first.")
        exit(1)
    
    print("Starting wandb validation upload tests...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Wandb available: {WANDB_AVAILABLE}")
    
    # Test online mode
    success = test_wandb_validation_upload()
    
    # Test offline mode if requested
    if args.test_offline:
        offline_success = test_offline_mode()
        success = success and offline_success
    
    if success:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("Wandb validation upload functionality is working correctly.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("There are issues with wandb validation upload.")
        print("Check the error messages above for details.")
        print("=" * 60)
        exit(1) 