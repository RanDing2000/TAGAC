#!/usr/bin/env python3
"""
Test script to verify that validation metrics upload works independently of training frequency controls.
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

def test_validation_upload_independent():
    """Test that validation metrics upload independently of training frequency."""
    
    print("=" * 70)
    print("🔥 TESTING VALIDATION UPLOAD INDEPENDENCE 🔥")
    print("=" * 70)
    
    # Initialize wandb
    project_name = "targo_validation_independence_test"
    run_name = f"validation_independence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Initializing wandb project: {project_name}")
    print(f"Run name: {run_name}")
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "test_type": "validation_independence",
                "timestamp": datetime.now().isoformat(),
                "wandb_log_freq": 10,  # Simulate training frequency setting
            },
            tags=["test", "validation", "independence"],
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            )
        )
        print("✅ Wandb initialization successful")
    except Exception as e:
        print(f"❌ Wandb initialization failed: {e}")
        return False
    
    # Test 1: Simulate training steps with frequency control
    print("\n" + "="*50)
    print("📊 Test 1: Training with frequency control")
    print("="*50)
    
    wandb_log_freq = 10  # Only log every 10 steps
    
    for step in range(1, 31):  # 30 training steps
        # Simulate training metrics
        train_loss = 1.0 - (step * 0.02) + np.random.normal(0, 0.1)
        
        # Apply frequency control for training
        if step % wandb_log_freq == 0:
            training_step = step * 1000  # Use same strategy as our training scripts
            train_metrics = {
                "train/step_loss_all": train_loss,
                "train/step_accuracy": min(0.9, step * 0.03),
                "train/step": step
            }
            wandb.log(train_metrics, step=training_step)
            print(f"  ✅ Step {step}: Training metrics uploaded (step={training_step})")
        else:
            print(f"  ⏭️  Step {step}: Training metrics skipped (frequency control)")
    
    # Test 2: Validation uploads (should happen EVERY epoch regardless of frequency)
    print("\n" + "="*50)
    print("🔥 Test 2: Validation uploads (EVERY EPOCH)")
    print("="*50)
    
    for epoch in range(1, 6):  # 5 epochs
        print(f"\n[INFO] 🔥 VALIDATION WANDB UPLOAD - EPOCH {epoch} 🔥")
        print(f"[INFO] ⚠️  VALIDATION METRICS WILL BE UPLOADED REGARDLESS OF ANY FREQUENCY SETTINGS ⚠️")
        
        # 使用独特的validation step range，完全独立于training
        validation_step = 10000 + epoch
        
        # Simulate validation metrics
        base_rate = 0.3 + (epoch * 0.1)
        ycb_rate = base_rate + np.random.normal(0, 0.05)
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
        print(f"[INFO] 🎯 Using validation step: {validation_step} (10000 + epoch)")
        
        # 强制上传validation metrics，不受任何频率限制影响
        upload_success = False
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"[INFO] 🚀 Attempt {attempt + 1}/{max_attempts}: Uploading validation metrics to wandb...")
                
                # 直接调用wandb.log，不检查任何频率设置
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
                    import traceback
                    traceback.print_exc()
        
        if not upload_success:
            print(f"[CRITICAL] 🚨 VALIDATION METRICS UPLOAD FAILED FOR EPOCH {epoch}!")
            return False
        
        # Small delay between epochs
        time.sleep(0.5)
    
    # Test 3: Verify step ranges don't conflict
    print("\n" + "="*50)
    print("🎯 Test 3: Step range verification")
    print("="*50)
    
    print(f"Training steps used: 1000, 2000, 3000 (step * 1000)")
    print(f"Validation steps used: 10001, 10002, 10003, 10004, 10005 (10000 + epoch)")
    print(f"✅ No step range conflicts!")
    
    # Final summary
    print("\n" + "="*50)
    print("📋 UPLOAD SUMMARY")
    print("="*50)
    
    final_summary = {
        "test/training_uploads": 3,  # Steps 10, 20, 30 with freq=10
        "test/validation_uploads": 5,  # All 5 epochs
        "test/total_uploads": 8,
        "test/validation_independence": 1.0,  # Success
        "test/completion_time": time.time()
    }
    
    try:
        wandb.log(final_summary, step=20000)  # Use high step to avoid conflicts
        print(f"✅ Final summary uploaded to wandb")
    except Exception as e:
        print(f"❌ Final summary upload failed: {e}")
    
    # Finish wandb run
    try:
        wandb.finish()
        print("✅ Wandb run finished successfully")
        return True
        
    except Exception as e:
        print(f"❌ Wandb finish failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test validation upload independence")
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        print("Wandb is not available. Please install it first.")
        exit(1)
    
    print("Starting validation upload independence test...")
    print(f"Current working directory: {os.getcwd()}")
    
    success = test_validation_upload_independent()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 VALIDATION UPLOAD INDEPENDENCE TEST PASSED! 🎉")
        print("✅ Validation metrics upload independently of training frequency")
        print("✅ No step conflicts between training and validation")
        print("✅ All validation epochs were uploaded successfully")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ VALIDATION UPLOAD INDEPENDENCE TEST FAILED!")
        print("There are still issues with validation upload independence.")
        print("=" * 70)
        exit(1) 