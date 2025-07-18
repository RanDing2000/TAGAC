#!/usr/bin/env python3
"""
下载GroundingDINO和SAM的模型权重文件
"""

import os
import sys
import requests
import urllib.request
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ {filename} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def main():
    print("Downloading GroundingDINO and SAM model weights...")
    print("=" * 60)
    
    # 模型下载链接
    models = {
        "groundingdino_swint_ogc.pth": {
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            "description": "GroundingDINO SwinT-OGC model"
        },
        "sam_vit_h_4b8939.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "description": "SAM ViT-H model"
        }
    }
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"Downloading to: {current_dir}")
    
    success_count = 0
    for filename, info in models.items():
        if os.path.exists(filename):
            print(f"✓ {filename} already exists, skipping...")
            success_count += 1
            continue
        
        print(f"\n{info['description']}")
        print(f"URL: {info['url']}")
        
        if download_file(info['url'], filename):
            success_count += 1
    
    print("\n" + "=" * 60)
    if success_count == len(models):
        print("✅ All model weights downloaded successfully!")
        print("\nYou can now run the demo:")
        print("python demo.py --text_prompt 'your prompt'")
    else:
        print(f"⚠ {success_count}/{len(models)} models downloaded successfully")
        print("Please check the error messages above and try downloading manually")
        
        print("\nManual download links:")
        for filename, info in models.items():
            if not os.path.exists(filename):
                print(f"- {filename}: {info['url']}")

if __name__ == "__main__":
    main() 