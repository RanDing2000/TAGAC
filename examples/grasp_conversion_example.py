#!/usr/bin/env python
"""
示例脚本：演示 AnyGrasp/FGC-Grasp 与 VGN 格式之间的转换
"""

import numpy as np
from scipy.spatial.transform import Rotation
from vgn.utils.transform import Transform
from vgn.grasp import Grasp
from vgn.grasp_conversion import anygrasp_to_vgn, vgn_to_anygrasp

# 如果安装了 graspnetAPI，则导入相关模块
try:
    from graspnetAPI.grasp import GraspGroup, Grasp as GraspnetGrasp
    GRASPNET_AVAILABLE = True
except ImportError:
    print("graspnetAPI not available. Parts of this example won't work.")
    print("To install: pip install graspnetAPI")
    GRASPNET_AVAILABLE = False


def create_sample_anygrasp_grasps():
    """创建示例 AnyGrasp 抓取，用于演示目的"""
    if not GRASPNET_AVAILABLE:
        print("Cannot create sample AnyGrasp grasps without graspnetAPI")
        return None
    
    grasp_list = []
    
    # 创建一些示例抓取
    for i in range(5):
        # 创建随机旋转矩阵
        r = Rotation.random()
        # 创建随机位置（在相机坐标系中）
        t = np.array([0.2, 0.1, 0.5]) + np.random.uniform(-0.1, 0.1, 3)
        # 随机抓取宽度
        width = np.random.uniform(0.02, 0.08)
        # 随机抓取得分
        score = np.random.uniform(0.5, 1.0)
        
        # 创建 GraspnetAPI 抓取
        grasp = GraspnetGrasp(
            rotation_matrix=r.as_matrix(),
            translation=t,
            width=width,
            score=score
        )
        grasp_list.append(grasp)
    
    return GraspGroup(grasp_list)


def create_sample_vgn_grasps():
    """创建示例 VGN 抓取，用于演示目的"""
    vgn_grasps = []
    scores = []
    
    # 创建一些示例抓取
    for i in range(5):
        # 创建随机旋转
        r = Rotation.random()
        # 创建随机位置（在世界坐标系中）
        t = np.array([0.3, 0.3, 0.1]) + np.random.uniform(-0.1, 0.1, 3)
        # 创建 Transform 对象
        pose = Transform(r, t)
        # 随机抓取宽度
        width = np.random.uniform(0.02, 0.08)
        
        # 创建 VGN 抓取
        grasp = Grasp(pose, width)
        vgn_grasps.append(grasp)
        scores.append(np.random.uniform(0.5, 1.0))
    
    return vgn_grasps, scores


def main():
    """主函数：演示 AnyGrasp 和 VGN 格式之间的转换"""
    print("Grasp Conversion Example")
    print("========================")
    
    # 创建示例相机外参矩阵（相机到世界的转换）
    # 假设相机在世界坐标系中位于 [0, 0, 1]，并向下俯视
    camera_rotation = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
    camera_translation = np.array([0, 0, 1])
    extrinsic = Transform(camera_rotation, camera_translation)
    
    print("\n1. 从 AnyGrasp 到 VGN 的转换")
    print("--------------------------")
    if GRASPNET_AVAILABLE:
        # 创建示例 AnyGrasp 抓取
        anygrasp_grasps = create_sample_anygrasp_grasps()
        print(f"创建了 {len(anygrasp_grasps)} 个 AnyGrasp 抓取")
        
        # 转换为 VGN 格式
        vgn_grasps, scores = anygrasp_to_vgn(anygrasp_grasps, extrinsic)
        print(f"转换为 {len(vgn_grasps)} 个 VGN 抓取")
        
        # 打印示例抓取信息
        if vgn_grasps:
            print("\n示例 VGN 抓取信息：")
            for i, grasp in enumerate(vgn_grasps[:2]):  # 只打印前两个
                print(f"抓取 {i+1}:")
                print(f"  位置: {grasp.pose.translation}")
                print(f"  旋转矩阵: \n{grasp.pose.rotation.as_matrix()}")
                print(f"  宽度: {grasp.width}")
                print(f"  得分: {scores[i]}")
    else:
        print("跳过 - 需要 graspnetAPI")
    
    print("\n2. 从 VGN 到 AnyGrasp 的转换")
    print("--------------------------")
    # 创建示例 VGN 抓取
    vgn_grasps, scores = create_sample_vgn_grasps()
    print(f"创建了 {len(vgn_grasps)} 个 VGN 抓取")
    
    if GRASPNET_AVAILABLE:
        # 转换为 AnyGrasp 格式
        anygrasp_grasps = vgn_to_anygrasp(vgn_grasps, scores, extrinsic)
        print(f"转换为 {len(anygrasp_grasps)} 个 AnyGrasp 抓取")
        
        # 打印示例抓取信息
        if len(anygrasp_grasps) > 0:
            print("\n示例 AnyGrasp 抓取信息：")
            for i, grasp in enumerate(anygrasp_grasps[:2]):  # 只打印前两个
                print(f"抓取 {i+1}:")
                print(f"  位置: {grasp.translation}")
                print(f"  旋转矩阵: \n{grasp.rotation_matrix}")
                print(f"  宽度: {grasp.width}")
                print(f"  得分: {grasp.score}")
    else:
        print("跳过 - 需要 graspnetAPI")
    
    print("\n3. 完整循环转换测试 (VGN -> AnyGrasp -> VGN)")
    print("------------------------------------------")
    if GRASPNET_AVAILABLE:
        # 创建示例 VGN 抓取
        original_vgn_grasps, original_scores = create_sample_vgn_grasps()
        
        # VGN -> AnyGrasp
        anygrasp_grasps = vgn_to_anygrasp(original_vgn_grasps, original_scores, extrinsic)
        
        # AnyGrasp -> VGN
        converted_vgn_grasps, converted_scores = anygrasp_to_vgn(anygrasp_grasps, extrinsic)
        
        print(f"原始 VGN 抓取数量: {len(original_vgn_grasps)}")
        print(f"转换后的 VGN 抓取数量: {len(converted_vgn_grasps)}")
        
        # 比较第一个抓取的位置和宽度
        if original_vgn_grasps and converted_vgn_grasps:
            original = original_vgn_grasps[0]
            converted = converted_vgn_grasps[0]
            
            print("\n比较第一个抓取:")
            print(f"原始位置: {original.pose.translation}")
            print(f"转换后位置: {converted.pose.translation}")
            print(f"位置差异: {np.linalg.norm(original.pose.translation - converted.pose.translation):.6f}")
            
            print(f"原始宽度: {original.width}")
            print(f"转换后宽度: {converted.width}")
            print(f"宽度差异: {abs(original.width - converted.width):.6f}")
    else:
        print("跳过 - 需要 graspnetAPI")
    
    print("\n完成示例")


if __name__ == "__main__":
    main() 