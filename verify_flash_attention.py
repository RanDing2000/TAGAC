#!/usr/bin/env python3
"""
Verification script to check if flash attention is enabled for PTv3 models.
"""

import sys
import os

def check_flash_attention_settings():
    """Check if flash attention is enabled in all PTv3 model configurations"""
    print("=" * 70)
    print("Flash Attention Verification for PTv3 Models")
    print("=" * 70)
    
    files_to_check = [
        ("src/transformer/ptv3_fusion_model.py", "PointTransformerV3FusionModel"),
        ("src/transformer/ptv3_scene_model.py", "PointTransformerV3SceneModel"),
        ("src/vgn/ConvONets/conv_onet/config.py", "get_model_ptv3_scene")
    ]
    
    all_enabled = True
    
    for file_path, model_name in files_to_check:
        print(f"\nChecking {model_name} in {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            all_enabled = False
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for enable_flash=True
        if "enable_flash=True" in content:
            print(f"✓ Flash attention is ENABLED")
            
            # Count occurrences
            true_count = content.count("enable_flash=True")
            false_count = content.count("enable_flash=False")
            
            print(f"  - enable_flash=True: {true_count} occurrences")
            print(f"  - enable_flash=False: {false_count} occurrences")
            
            if false_count > 0:
                print(f"  ⚠ Warning: Still has {false_count} instances of enable_flash=False")
                all_enabled = False
        else:
            print(f"✗ Flash attention is NOT enabled (no enable_flash=True found)")
            all_enabled = False
    
    # Check if flash_attn is available
    print(f"\nChecking flash_attn library availability...")
    try:
        import flash_attn
        print("✓ flash_attn library is available")
        print(f"  Version: {getattr(flash_attn, '__version__', 'unknown')}")
    except ImportError:
        print("✗ flash_attn library is NOT available")
        print("  Install with: pip install flash-attn")
        all_enabled = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary:")
    if all_enabled:
        print("✓ Flash attention is ENABLED for all PTv3 models")
        print("✓ All configurations are optimized for performance")
        print("\nBenefits of flash attention:")
        print("- Faster attention computation")
        print("- Reduced memory usage")
        print("- Better training and inference speed")
    else:
        print("✗ Flash attention configuration issues detected")
        print("Please review the files and ensure enable_flash=True is set correctly")
    
    print("=" * 70)
    
    return all_enabled

def check_model_imports():
    """Check if PTv3 models can be imported successfully"""
    print("\nTesting model imports...")
    
    try:
        # Test PTv3 fusion model import
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.transformer.ptv3_fusion_model import PointTransformerV3FusionModel
        print("✓ PointTransformerV3FusionModel imported successfully")
        
        # Test PTv3 scene model import
        from src.transformer.ptv3_scene_model import PointTransformerV3SceneModel
        print("✓ PointTransformerV3SceneModel imported successfully")
        
        # Test config function import
        from src.vgn.ConvONets.conv_onet.config import get_model_ptv3_scene
        print("✓ get_model_ptv3_scene function imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Main verification function"""
    # Check flash attention settings
    flash_ok = check_flash_attention_settings()
    
    # Check model imports
    import_ok = check_model_imports()
    
    # Final status
    print(f"\nFinal Status:")
    print(f"Flash attention settings: {'✓ OK' if flash_ok else '✗ ISSUES'}")
    print(f"Model imports: {'✓ OK' if import_ok else '✗ ISSUES'}")
    
    if flash_ok and import_ok:
        print(f"\n🎉 All checks passed! Flash attention is ready to use.")
        print(f"\nNext steps:")
        print(f"1. Load CUDA modules: module load compiler/gcc-8.3 && module load cuda/11.3.0")
        print(f"2. Activate environment: conda activate targo")
        print(f"3. Run training with optimized models:")
        print(f"   - TARGO PTv3: python scripts/train_targo_ptv3.py --net targo_ptv3")
        print(f"   - PTv3 Scene: python scripts/train_targo_ptv3.py --net ptv3_scene")
    else:
        print(f"\n❌ Some issues detected. Please review and fix before proceeding.")

if __name__ == "__main__":
    main() 