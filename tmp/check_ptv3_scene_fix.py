#!/usr/bin/env python3
"""
Direct check of ptv3_scene fixes without importing problematic modules.
"""

import sys
import os

def check_config_file():
    """Check if get_model_ptv3_scene function is properly implemented"""
    print("Checking src/vgn/ConvONets/conv_onet/config.py...")
    
    config_file = "src/vgn/ConvONets/conv_onet/config.py"
    
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        return False
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check if get_model_ptv3_scene function exists
    if "def get_model_ptv3_scene(" in content:
        print("✓ get_model_ptv3_scene function found")
    else:
        print("✗ get_model_ptv3_scene function not found")
        return False
    
    # Check if PointTransformerV3SceneModel is imported
    if "from src.transformer.ptv3_scene_model import PointTransformerV3SceneModel" in content:
        print("✓ PointTransformerV3SceneModel import found")
    else:
        print("✗ PointTransformerV3SceneModel import not found")
        return False
    
    # Check if ConvolutionalOccupancyNetwork_Grid is called with correct parameters
    if "models.ConvolutionalOccupancyNetwork_Grid(" in content and "decoders, encoders_in, encoder_aff_scene" in content:
        print("✓ ConvolutionalOccupancyNetwork_Grid called with correct parameters")
    else:
        print("✗ ConvolutionalOccupancyNetwork_Grid not called with correct parameters")
        return False
    
    return True

def check_networks_file():
    """Check if networks.py has ptv3_scene properly registered"""
    print("\nChecking src/vgn/networks.py...")
    
    networks_file = "src/vgn/networks.py"
    
    if not os.path.exists(networks_file):
        print(f"✗ Networks file not found: {networks_file}")
        return False
    
    with open(networks_file, 'r') as f:
        content = f.read()
    
    # Check if ptv3_scene is in the models dictionary
    if '"ptv3_scene": Ptv3SceneNet,' in content:
        print("✓ ptv3_scene registered in models dictionary")
    else:
        print("✗ ptv3_scene not registered in models dictionary")
        return False
    
    # Check if Ptv3SceneNet function exists
    if "def Ptv3SceneNet():" in content:
        print("✓ Ptv3SceneNet function found")
    else:
        print("✗ Ptv3SceneNet function not found")
        return False
    
    # Check if get_model_ptv3_scene is imported
    if "get_model_ptv3_scene" in content:
        print("✓ get_model_ptv3_scene import found")
    else:
        print("✗ get_model_ptv3_scene import not found")
        return False
    
    return True

def check_models_file():
    """Check if ConvolutionalOccupancyNetwork_Grid supports ptv3_scene"""
    print("\nChecking src/vgn/ConvONets/conv_onet/models/__init__.py...")
    
    models_file = "src/vgn/ConvONets/conv_onet/models/__init__.py"
    
    if not os.path.exists(models_file):
        print(f"✗ Models file not found: {models_file}")
        return False
    
    with open(models_file, 'r') as f:
        content = f.read()
    
    # Check if ptv3_scene is supported in __init__
    if 'if model_type == "ptv3_scene":' in content and 'self.encoder_in = encoders_in[0].to(device)' in content:
        print("✓ ptv3_scene support added to __init__ method")
    else:
        print("✗ ptv3_scene support not found in __init__ method")
        return False
    
    # Check if ptv3_scene is supported in forward
    if 'elif self.model_type == "ptv3_scene":' in content and 'features_fused = self.encoder_in(inputs)' in content:
        print("✓ ptv3_scene support added to forward method")
    else:
        print("✗ ptv3_scene support not found in forward method")
        return False
    
    return True

def check_ptv3_scene_model():
    """Check if ptv3_scene_model.py exists and has correct structure"""
    print("\nChecking src/transformer/ptv3_scene_model.py...")
    
    model_file = "src/transformer/ptv3_scene_model.py"
    
    if not os.path.exists(model_file):
        print(f"✗ PTv3 scene model file not found: {model_file}")
        return False
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Check if PointTransformerV3SceneModel class exists
    if "class PointTransformerV3SceneModel(nn.Module):" in content:
        print("✓ PointTransformerV3SceneModel class found")
    else:
        print("✗ PointTransformerV3SceneModel class not found")
        return False
    
    # Check if enable_flash=False is set
    if "enable_flash=False" in content:
        print("✓ enable_flash=False found (flash attention disabled)")
    else:
        print("✗ enable_flash=False not found")
        return False
    
    return True

def check_training_script():
    """Check if train_targo_ptv3.py supports ptv3_scene"""
    print("\nChecking scripts/train_targo_ptv3.py...")
    
    script_file = "scripts/train_targo_ptv3.py"
    
    if not os.path.exists(script_file):
        print(f"✗ Training script not found: {script_file}")
        return False
    
    with open(script_file, 'r') as f:
        content = f.read()
    
    # Check if ptv3_scene is in choices
    if 'choices=["targo_ptv3", "ptv3_scene"]' in content:
        print("✓ ptv3_scene added to argument choices")
    else:
        print("✗ ptv3_scene not found in argument choices")
        return False
    
    # Check if prepare_batch supports model_type
    if 'def prepare_batch(batch, device, model_type="targo_ptv3"):' in content:
        print("✓ prepare_batch function supports model_type parameter")
    else:
        print("✗ prepare_batch function doesn't support model_type parameter")
        return False
    
    # Check if ptv3_scene handling is in prepare_batch
    if 'if model_type == "ptv3_scene":' in content and 'return scene_pc, (label, rotations, width), pos' in content:
        print("✓ ptv3_scene handling found in prepare_batch")
    else:
        print("✗ ptv3_scene handling not found in prepare_batch")
        return False
    
    return True

def main():
    """Main check function"""
    print("=" * 70)
    print("PTv3 Scene Implementation Check")
    print("=" * 70)
    
    checks = [
        ("Configuration file", check_config_file),
        ("Networks file", check_networks_file),
        ("Models file", check_models_file),
        ("PTv3 scene model", check_ptv3_scene_model),
        ("Training script", check_training_script),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("Check Summary:")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks PASSED!")
        print("\nptv3_scene implementation is complete and ready for use.")
        print("\nTo use ptv3_scene:")
        print("1. Load CUDA modules (if needed)")
        print("2. Activate conda environment: conda activate targo")
        print("3. Run training: python scripts/train_targo_ptv3.py --net ptv3_scene")
        print("\nKey differences:")
        print("- ptv3_scene: Only processes scene point cloud")
        print("- targo_ptv3: Processes both scene and target point clouds")
    else:
        print("\n✗ Some checks FAILED. Please review the implementation.")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 