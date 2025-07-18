#!/usr/bin/env python3
"""
测试GroundingDINO和SAM的导入是否正常
"""

import os
import sys

def test_imports():
    print("Testing GroundingDINO and SAM imports...")
    print("=" * 50)
    
    # 测试GroundingDINO导入
    print("1. Testing GroundingDINO import...")
    try:
        # 添加GroundingDINO路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        groundingdino_path = os.path.join(current_dir, '../../src/GroundingDINO')
        sys.path.append(groundingdino_path)
        
        from groundingdino.util.inference import load_model, load_image, predict
        print("   ✓ GroundingDINO imported successfully")
        
        # 检查配置文件
        config_path = os.path.join(groundingdino_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        if os.path.exists(config_path):
            print("   ✓ GroundingDINO config file found")
        else:
            print("   ✗ GroundingDINO config file not found")
            
    except ImportError as e:
        print(f"   ✗ Failed to import GroundingDINO: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error with GroundingDINO: {e}")
        return False
    
    # 测试SAM导入
    print("\n2. Testing SAM import...")
    try:
        # 添加SAM路径
        sam_path = os.path.join(current_dir, '../../src/segment-anything')
        sys.path.append(sam_path)
        
        from segment_anything import sam_model_registry, SamPredictor
        print("   ✓ SAM imported successfully")
        
        # 检查SAM文件
        sam_init_path = os.path.join(sam_path, "segment_anything/__init__.py")
        if os.path.exists(sam_init_path):
            print("   ✓ SAM module files found")
        else:
            print("   ✗ SAM module files not found")
            
    except ImportError as e:
        print(f"   ✗ Failed to import SAM: {e}")
        print("   Please make sure SAM is installed in: " + sam_path)
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error with SAM: {e}")
        return False
    
    # 测试其他依赖
    print("\n3. Testing other dependencies...")
    try:
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"   ✗ PyTorch not found: {e}")
        return False
    
    try:
        import cv2
        print(f"   ✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"   ✗ OpenCV not found: {e}")
        return False
    
    try:
        import matplotlib
        print(f"   ✓ Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"   ✗ Matplotlib not found: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"   ✗ NumPy not found: {e}")
        return False
    
    # 检查CUDA可用性
    print("\n4. Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠ CUDA not available - will use CPU (slower)")
    
    # 检查模型权重文件
    print("\n5. Checking model weight files...")
    dino_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    
    if os.path.exists(dino_checkpoint):
        print(f"   ✓ GroundingDINO checkpoint found: {dino_checkpoint}")
    else:
        print(f"   ⚠ GroundingDINO checkpoint not found: {dino_checkpoint}")
        print("   Please download from: https://github.com/IDEA-Research/GroundingDINO/releases")
    
    if os.path.exists(sam_checkpoint):
        print(f"   ✓ SAM checkpoint found: {sam_checkpoint}")
    else:
        print(f"   ⚠ SAM checkpoint not found: {sam_checkpoint}")
        print("   Please download from: https://github.com/facebookresearch/segment-anything")
    
    # 检查默认图像文件
    print("\n6. Checking default image file...")
    default_image = "/usr/stud/dira/GraspInClutter/targo/scripts/demo/image.png"
    if os.path.exists(default_image):
        print(f"   ✓ Default image found: {default_image}")
    else:
        print(f"   ⚠ Default image not found: {default_image}")
        print("   Please provide an image path when running the demo")
    
    print("\n" + "=" * 50)
    print("Import test completed successfully!")
    print("You can now run the demo with:")
    print("python demo.py --image_path your_image.png --text_prompt 'your prompt'")
    print("Or use the default image:")
    print("python demo.py --text_prompt 'your prompt'")
    
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n❌ Import test failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All imports successful!") 