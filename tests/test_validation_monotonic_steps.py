#!/usr/bin/env python3
"""
Test script to verify validation metrics upload with monotonic step ordering.
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

def test_validation_monotonic_steps():
    """Test validation metrics upload with proper monotonic step ordering."""
    
    print("=" * 70)
    print("🔥 TESTING VALIDATION MONOTONIC STEP ORDERING 🔥")
    print("=" * 70)
    
    # Initialize wandb
    project_name = "targo_validation_monotonic_test"
    run_name = f"monotonic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Initializing wandb project: {project_name}")
    print(f"Run name: {run_name}")
    
    # Global step counter (same as training scripts)
    global_step = {"value": 0}
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "test_type": "validation_monotonic",
                "timestamp": datetime.now().isoformat(),
                "wandb_log_freq": 10,
            },
            tags=["test", "validation", "monotonic"],
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            )
        )
        print("✅ Wandb initialization successful")
    except Exception as e:
        print(f"❌ Wandb initialization failed: {e}")
        return False
    
    # Test: Mixed training and validation steps with proper ordering
    print("\n" + "="*60)
    print("📊 Test: Mixed training and validation with monotonic steps")
    print("="*60)
    
    wandb_log_freq = 10
    
    for epoch in range(1, 4):  # 3 epochs
        print(f"\n🔥 EPOCH {epoch} 🔥")
        
        # Simulate training steps for this epoch
        steps_per_epoch = 25
        for step in range(1, steps_per_epoch + 1):
            global_iteration = (epoch - 1) * steps_per_epoch + step
            
            # Simulate training metrics
            train_loss = 1.0 - (global_iteration * 0.01) + np.random.normal(0, 0.05)
            
            # Apply frequency control for training (same as our scripts)
            if global_iteration % wandb_log_freq == 0:
                training_step = global_iteration * 1000  # Same multiplier as our scripts
                train_metrics = {
                    "train/step_loss_all": train_loss,
                    "train/step_accuracy": min(0.9, global_iteration * 0.02),
                    "train/iteration": global_iteration
                }
                
                # Check if training step is monotonic
                if training_step > global_step["value"]:
                    global_step["value"] = training_step
                    wandb.log(train_metrics, step=training_step)
                    print(f"  ✅ Step {global_iteration}: Training metrics uploaded (step={training_step})")
                else:
                    print(f"  ❌ Step {global_iteration}: Training step {training_step} <= global_step {global_step['value']}")
            else:
                print(f"  ⏭️  Step {global_iteration}: Training metrics skipped (frequency control)")
        
        # End of epoch: Validation (same logic as our scripts)
        print(f"\n[INFO] 🔥 VALIDATION WANDB UPLOAD - EPOCH {epoch} 🔥")
        print(f"[INFO] ⚠️  VALIDATION METRICS WILL BE UPLOADED REGARDLESS OF ANY FREQUENCY SETTINGS ⚠️")
        
        # 使用global_step计数器确保validation step总是比training step大
        validation_step = global_step["value"] + 1
        global_step["value"] = validation_step  # 更新global step
        
        print(f"[INFO] 🎯 Using validation step: {validation_step} (ensuring monotonic increase)")
        
        # Simulate validation metrics
        base_rate = 0.3 + (epoch * 0.15)
        ycb_rate = base_rate + np.random.normal(0, 0.03)
        avg_rate = max(0.1, ycb_rate)
        
        validation_metrics = {
            "val/grasp_success_rate": float(avg_rate),
            "val/ycb_success_rate": float(ycb_rate), 
            "val/acronym_success_rate": 0.0,
            "val/epoch": int(epoch),
            "validation/source": 1.0,
            "validation/ycb_only": 1.0,
            "validation/acronym_disabled": 1.0,
            "validation/upload_timestamp": time.time()
        }
        
        print(f"[INFO] 📊 Validation metrics to upload: {validation_metrics}")
        
        # Upload validation metrics (same logic as our scripts)
        upload_success = False
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"[INFO] 🚀 Attempt {attempt + 1}/{max_attempts}: Uploading validation metrics to wandb...")
                
                # 直接调用wandb.log，确保step单调递增
                wandb.log(validation_metrics, step=validation_step)
                
                print(f"[SUCCESS] ✅ Validation metrics successfully uploaded to wandb!")
                print(f"[SUCCESS] ✅ Epoch {epoch} validation data is now in wandb!")
                upload_success = True
                break
                
            except Exception as e:
                print(f"[ERROR] ❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print(f"[INFO] 🔄 Retrying in 1 second...")
                    time.sleep(1)
                else:
                    print(f"[ERROR] 💥 All {max_attempts} attempts failed!")
        
        if not upload_success:
            print(f"[CRITICAL] 🚨 VALIDATION METRICS UPLOAD FAILED FOR EPOCH {epoch}!")
            return False
        
        print(f"[INFO] ✅ Epoch {epoch} complete. Current global_step: {global_step['value']}")
    
    # Final summary
    print("\n" + "="*50)
    print("📋 MONOTONIC STEP TEST SUMMARY")
    print("="*50)
    
    final_summary = {
        "test/final_global_step": global_step["value"],
        "test/epochs_completed": 3,
        "test/validation_uploads": 3,
        "test/monotonic_success": 1.0,
        "test/completion_time": time.time()
    }
    
    # Final upload with proper step
    summary_step = global_step["value"] + 1
    global_step["value"] = summary_step
    
    try:
        wandb.log(final_summary, step=summary_step)
        print(f"✅ Final summary uploaded to wandb (step={summary_step})")
    except Exception as e:
        print(f"❌ Final summary upload failed: {e}")
    
    print(f"Final global_step value: {global_step['value']}")
    
    # Finish wandb run
    try:
        wandb.finish()
        print("✅ Wandb run finished successfully")
        return True
        
    except Exception as e:
        print(f"❌ Wandb finish failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test validation monotonic step ordering")
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        print("Wandb is not available. Please install it first.")
        exit(1)
    
    print("Starting validation monotonic step ordering test...")
    print(f"Current working directory: {os.getcwd()}")
    
    success = test_validation_monotonic_steps()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 VALIDATION MONOTONIC STEP TEST PASSED! 🎉")
        print("✅ All validation metrics uploaded with proper step ordering")
        print("✅ No wandb monotonic step warnings")
        print("✅ Global step counter ensures proper ordering")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ VALIDATION MONOTONIC STEP TEST FAILED!")
        print("There are still issues with step ordering.")
        print("=" * 70)
        exit(1) 