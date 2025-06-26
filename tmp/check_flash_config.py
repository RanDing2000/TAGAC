#!/usr/bin/env python3
"""
Simple script to check flash attention configuration in PTv3 models.
"""

import os

def check_flash_attention_config():
    """Check if flash attention is enabled in configuration files"""
    print("=" * 60)
    print("Flash Attention Configuration Check")
    print("=" * 60)
    
    files_to_check = [
        ("src/transformer/ptv3_fusion_model.py", "PointTransformerV3FusionModel"),
        ("src/transformer/ptv3_scene_model.py", "PointTransformerV3SceneModel"),
        ("src/vgn/ConvONets/conv_onet/config.py", "get_model_ptv3_scene")
    ]
    
    all_enabled = True
    
    for file_path, model_name in files_to_check:
        print(f"\n{model_name}:")
        print(f"File: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"✗ File not found")
            all_enabled = False
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Count occurrences
        true_count = content.count("enable_flash=True")
        false_count = content.count("enable_flash=False")
        
        print(f"enable_flash=True: {true_count}")
        print(f"enable_flash=False: {false_count}")
        
        if true_count > 0 and false_count == 0:
            print("✓ Flash attention ENABLED")
        elif true_count > 0 and false_count > 0:
            print("⚠ Mixed settings detected")
            all_enabled = False
        else:
            print("✗ Flash attention NOT enabled")
            all_enabled = False
    
    print("\n" + "=" * 60)
    print("Summary:")
    if all_enabled:
        print("✓ Flash attention is ENABLED for all PTv3 models")
        print("\nBenefits:")
        print("- Faster attention computation")
        print("- Reduced memory usage")
        print("- Better performance")
        print("\nNote: flash_attn library must be installed for this to work")
        print("Install with: pip install flash-attn")
    else:
        print("✗ Configuration issues detected")
    
    print("=" * 60)
    
    return all_enabled

if __name__ == "__main__":
    check_flash_attention_config() 