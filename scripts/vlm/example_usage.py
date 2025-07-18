#!/usr/bin/env python3
"""
GroundingDINO + SAM Demo 使用示例
"""

import os
import subprocess
import sys

def run_demo_command(cmd, description):
    """运行demo命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"示例: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 命令执行成功")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print("✗ 命令执行失败")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
    except Exception as e:
        print(f"✗ 执行命令时出错: {e}")

def main():
    print("GroundingDINO + SAM Demo 使用示例")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查必要文件
    print("\n检查必要文件...")
    required_files = [
        "demo.py",
        "test_imports.py", 
        "download_models.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} 不存在")
            return
    
    # 示例1: 测试环境
    print("\n1. 测试环境配置")
    run_demo_command("python test_imports.py", "测试所有依赖是否正确安装")
    
    # 示例2: 下载模型
    print("\n2. 下载模型权重")
    run_demo_command("python download_models.py", "自动下载GroundingDINO和SAM模型权重")
    
    # 示例3: 基本使用
    print("\n3. 基本使用示例")
    run_demo_command("python demo.py --text_prompt 'person'", "使用默认图像检测人物")
    
    # 示例4: 自定义参数
    print("\n4. 自定义参数示例")
    run_demo_command("python demo.py --text_prompt 'object' --box_threshold 0.2 --text_threshold 0.2", "使用较低阈值检测对象")
    
    # 示例5: 帮助信息
    print("\n5. 查看帮助信息")
    run_demo_command("python demo.py --help", "显示所有可用参数")
    
    print("\n" + "="*60)
    print("示例执行完成！")
    print("\n更多使用示例:")
    print("- 检测特定物体: python demo.py --text_prompt 'car'")
    print("- 检测多个物体: python demo.py --text_prompt 'chair and table'")
    print("- 使用自定义图像: python demo.py --image_path your_image.png --text_prompt 'your prompt'")
    print("- 调整输出路径: python demo.py --text_prompt 'person' --output_path my_result.png")

if __name__ == "__main__":
    main() 