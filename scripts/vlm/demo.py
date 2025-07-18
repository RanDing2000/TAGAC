import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import argparse

# Add GroundingDINO to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
groundingdino_path = os.path.join(current_dir, '../../src/GroundingDINO')
sys.path.append(groundingdino_path)

# Add SAM to Python path
sam_path = os.path.join(current_dir, '../../src/segment-anything')
sys.path.append(sam_path)

# ========== Grounding DINO ==========
try:
    from groundingdino.util.inference import load_model, load_image, predict
    print("✓ GroundingDINO imported successfully")
    
    # 检查_C模块
    try:
        import groundingdino._C
        print("✓ GroundingDINO C++ extension available")
    except ImportError:
        print("⚠ GroundingDINO C++ extension not available, using CPU fallback")
        print("   For better performance, run: python fix_groundingdino.py")
        
except ImportError as e:
    print(f"✗ Failed to import GroundingDINO: {e}")
    print(f"Make sure GroundingDINO is installed in: {groundingdino_path}")
    print("Try running: python fix_groundingdino.py")
    sys.exit(1)

# ========== SAM ==========
try:
    from segment_anything import sam_model_registry, SamPredictor
    print("✓ SAM imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SAM: {e}")
    print(f"Make sure SAM is installed in: {sam_path}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='GroundingDINO + SAM Demo')
    parser.add_argument('--image_path', type=str, default="/usr/stud/dira/GraspInClutter/targo/scripts/demo/image.png", 
                       help='Path to input image')
    parser.add_argument('--text_prompt', type=str, default="black figure", 
                       help='Text prompt for object detection')
    parser.add_argument('--output_path', type=str, default="output_masked.png", 
                       help='Path to save output image')
    parser.add_argument('--dino_checkpoint', type=str, default="/usr/stud/dira/GraspInClutter/targo/src/GroundingDINO/groundingdino_swint_ogc.pth", 
                       help='Path to GroundingDINO checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, default="/usr/stud/dira/GraspInClutter/targo/src/GroundingDINO/sam_vit_h_4b8939.pth", 
                       help='Path to SAM checkpoint')
    parser.add_argument('--box_threshold', type=float, default=0.3, 
                       help='Box confidence threshold')
    parser.add_argument('--text_threshold', type=float, default=0.25, 
                       help='Text confidence threshold')
    parser.add_argument('--device', type=str, default="auto", 
                       help='Device to use (auto, cuda, cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ========== 配置路径 ==========
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_path = args.output_path
    
    # 模型路径 - 使用相对路径
    dino_config = os.path.join(groundingdino_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    dino_checkpoint = args.dino_checkpoint
    sam_checkpoint = args.sam_checkpoint
    
    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # 验证文件存在
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        print("Please provide a valid image path or place an image at the default location")
        sys.exit(1)
    
    if not os.path.exists(dino_config):
        print(f"✗ GroundingDINO config not found: {dino_config}")
        sys.exit(1)
    
    if not os.path.exists(dino_checkpoint):
        print(f"✗ GroundingDINO checkpoint not found: {dino_checkpoint}")
        print("Please download the checkpoint file and place it in the specified location")
        print("Download from: https://github.com/IDEA-Research/GroundingDINO/releases")
        sys.exit(1)
    
    if not os.path.exists(sam_checkpoint):
        print(f"✗ SAM checkpoint not found: {sam_checkpoint}")
        print("Please download the SAM checkpoint file and place it in the specified location")
        print("Download from: https://github.com/facebookresearch/segment-anything")
        sys.exit(1)
    
    print(f"✓ All files found successfully")
    print(f"Image: {image_path}")
    print(f"Text prompt: {text_prompt}")
    print(f"Output: {output_path}")
    
    # ========== 加载 GroundingDINO 模型 ==========
    print("\nLoading GroundingDINO...")
    try:
        dino_model = load_model(dino_config, dino_checkpoint, device=device)
        image_source, image = load_image(image_path)
        print("✓ GroundingDINO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load GroundingDINO model: {e}")
        print("This might be due to missing C++ extension. Try running: python fix_groundingdino.py")
        sys.exit(1)
    
    # ========== GroundingDINO 预测 ==========
    print("Running GroundingDINO prediction...")
    try:
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device
        )
        print(f"✓ Found {len(boxes)} objects")
        for i, (phrase, logit) in enumerate(zip(phrases, logits)):
            print(f"  {i+1}. {phrase} (confidence: {logit:.3f})")
    except Exception as e:
        print(f"✗ GroundingDINO prediction failed: {e}")
        print("This might be due to missing C++ extension. Try running: python fix_groundingdino.py")
        sys.exit(1)
    
    if len(boxes) == 0:
        print("No objects detected. Try adjusting thresholds or changing the text prompt.")
        return
    
    # ========== 加载 SAM ==========
    print("\nLoading SAM...")
    try:
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        if device == "cuda":
            sam = sam.cuda()
        predictor = SamPredictor(sam)
        print("✓ SAM model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load SAM model: {e}")
        sys.exit(1)
    
    # 读取图像
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    # 尺寸转换 box 并生成 mask
    H, W, _ = image_rgb.shape
    boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
    boxes_xyxy = boxes_xyxy.cpu().numpy()
    
    print("Generating masks with SAM...")
    masks = []
    for i, box in enumerate(boxes_xyxy):
        try:
            masks_single, _, _ = predictor.predict(box=box, multimask_output=False)
            masks.append(masks_single[0])
            print(f"✓ Generated mask {i+1}/{len(boxes_xyxy)}")
        except Exception as e:
            print(f"✗ Failed to generate mask {i+1}: {e}")
            continue
    
    if len(masks) == 0:
        print("No masks generated. Check if the detected boxes are valid.")
        return
    
    # ========== 保存叠加结果 ==========
    print(f"\nSaving result to {output_path}...")
    try:
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        
        # 使用不同颜色显示不同的mask
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i, m in enumerate(masks):
            color = colors[i % len(colors)]
            plt.imshow(m, alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]))
        
        plt.title(f"Prompt: {text_prompt}\nDetected objects: {', '.join(phrases)}")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        print(f"✓ Result saved successfully to {output_path}")
    except Exception as e:
        print(f"✗ Failed to save result: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
