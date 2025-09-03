#!/usr/bin/env python3
"""
修复GLB文件中的颜色因子问题
"""

import os
import numpy as np
import trimesh
import sys
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

def fix_glb_colors(glb_path, output_path=None):
    """修复GLB文件中的颜色因子问题"""
    
    if output_path is None:
        base_name = os.path.splitext(glb_path)[0]
        output_path = f"{base_name}_fixed.glb"
    
    print(f"修复GLB文件: {glb_path}")
    print(f"输出文件: {output_path}")
    
    # 加载GLB文件
    mesh = trimesh.load(glb_path, process=False)
    
    if isinstance(mesh, trimesh.Scene):
        print(f"场景中的几何体数量: {len(mesh.geometry)}")
        
        for name, geom in mesh.geometry.items():
            print(f"\n处理几何体: {name}")
            
            if hasattr(geom, 'visual') and geom.visual is not None:
                if hasattr(geom.visual, 'material'):
                    material = geom.visual.material
                    
                    # 检查基础颜色因子
                    if hasattr(material, 'baseColorFactor'):
                        original_factor = material.baseColorFactor
                        print(f"  原始基础颜色因子: {original_factor}")
                        
                        # 如果基础颜色因子是纯白色，将其设置为None或[1,1,1,1]
                        if original_factor is not None and np.all(original_factor == [255, 255, 255, 255]):
                            print(f"  检测到纯白色基础颜色因子，正在修复...")
                            # 方法1: 设置为None（让纹理颜色生效）
                            material.baseColorFactor = None
                            print(f"  修复后基础颜色因子: {material.baseColorFactor}")
                        elif original_factor is not None and np.all(original_factor == [1, 1, 1, 1]):
                            print(f"  检测到单位白色基础颜色因子，正在修复...")
                            material.baseColorFactor = None
                            print(f"  修复后基础颜色因子: {material.baseColorFactor}")
                        else:
                            print(f"  基础颜色因子正常，无需修复")
                    
                    # 检查是否有纹理
                    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                        print(f"  基础颜色纹理: {material.baseColorTexture}")
                        print(f"  纹理模式: {material.baseColorTexture.mode}")
                        print(f"  纹理尺寸: {material.baseColorTexture.size}")
                    else:
                        print(f"  无基础颜色纹理")
    
    # 保存修复后的GLB文件
    mesh.export(output_path)
    print(f"\n修复后的GLB文件已保存: {output_path}")
    
    return output_path

def test_render_comparison(original_glb, fixed_glb, output_dir="render_comparison"):
    """测试渲染对比"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== 渲染对比测试 ===")
    
    # 导入渲染函数
    from src.utils.data_utils import normalize_mesh
    from src.utils.render_utils import render_single_view
    
    # 渲染原始GLB
    print("渲染原始GLB...")
    original_mesh = trimesh.load(original_glb, process=False)
    original_mesh_normalized = normalize_mesh(original_mesh)
    original_mesh_geometry = original_mesh_normalized.to_geometry()
    
    original_image = render_single_view(
        original_mesh_geometry,
        radius=4,
        image_size=(1024, 1024),
        light_intensity=1.5,
        num_env_lights=36,
        return_type='pil'
    )
    original_image.save(os.path.join(output_dir, 'original_render.png'))
    
    # 渲染修复后的GLB
    print("渲染修复后的GLB...")
    fixed_mesh = trimesh.load(fixed_glb, process=False)
    fixed_mesh_normalized = normalize_mesh(fixed_mesh)
    fixed_mesh_geometry = fixed_mesh_normalized.to_geometry()
    
    fixed_image = render_single_view(
        fixed_mesh_geometry,
        radius=4,
        image_size=(1024, 1024),
        light_intensity=1.5,
        num_env_lights=36,
        return_type='pil'
    )
    fixed_image.save(os.path.join(output_dir, 'fixed_render.png'))
    
    print(f"渲染对比结果保存至: {output_dir}/")
    
    return output_dir

def main():
    # 修复messy_kitchen GLB文件
    messy_kitchen_glb = "assets/messy_kitchen/5f8584bf10dd4bbb9651142fd37e73fa_combined.glb"
    
    if os.path.exists(messy_kitchen_glb):
        # 修复GLB文件
        fixed_glb = fix_glb_colors(messy_kitchen_glb)
        
        # 测试渲染对比
        test_render_comparison(messy_kitchen_glb, fixed_glb)
        
        print(f"\n=== 修复完成 ===")
        print(f"原始文件: {messy_kitchen_glb}")
        print(f"修复文件: {fixed_glb}")
        print(f"请使用修复后的GLB文件进行渲染")
    else:
        print(f"文件不存在: {messy_kitchen_glb}")

if __name__ == "__main__":
    main()
