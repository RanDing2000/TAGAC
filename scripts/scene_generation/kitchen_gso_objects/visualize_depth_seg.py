#!/usr/bin/env python3
"""
Script to visualize depth maps and instance segmentation maps from generated scenes.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import json
import glob

def visualize_depth_map(depth_img, output_path, scene_id, prefix="depth"):
    """
    Visualize depth map and save as image.
    
    Parameters:
    - depth_img: 2D numpy array containing depth values
    - output_path: Path to save the visualization
    - scene_id: Scene identifier
    - prefix: Prefix for the filename
    """
    try:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Create depth visualization
        plt.subplot(2, 3, 1)
        plt.imshow(depth_img, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title(f'Depth Map - {scene_id}')
        plt.axis('off')
        
        # Create depth histogram
        plt.subplot(2, 3, 2)
        valid_depths = depth_img[depth_img > 0]
        if len(valid_depths) > 0:
            plt.hist(valid_depths, bins=50, alpha=0.7, color='blue')
            plt.xlabel('Depth (m)')
            plt.ylabel('Frequency')
            plt.title('Depth Distribution')
        
        # Create depth mask (valid/invalid pixels)
        plt.subplot(2, 3, 3)
        depth_mask = (depth_img > 0).astype(np.uint8)
        plt.imshow(depth_mask, cmap='gray')
        plt.title('Valid Depth Mask')
        plt.axis('off')
        
        # Create depth with different color scheme
        plt.subplot(2, 3, 4)
        plt.imshow(depth_img, cmap='jet')
        plt.colorbar(label='Depth (m)')
        plt.title('Depth Map (Jet Colormap)')
        plt.axis('off')
        
        # Depth statistics
        plt.subplot(2, 3, 5)
        if len(valid_depths) > 0:
            stats_text = f"""
            Depth Statistics:
            Min: {depth_img.min():.3f} m
            Max: {depth_img.max():.3f} m
            Mean: {depth_img.mean():.3f} m
            Valid pixels: {len(valid_depths)}
            Total pixels: {depth_img.size}
            """
            plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.axis('off')
        plt.title('Depth Statistics')
        
        # Depth with log scale
        plt.subplot(2, 3, 6)
        if len(valid_depths) > 0:
            log_depth = np.log1p(depth_img)
            plt.imshow(log_depth, cmap='plasma')
            plt.colorbar(label='Log(Depth + 1)')
            plt.title('Depth Map (Log Scale)')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        filename = f"{prefix}_{scene_id}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved depth visualization: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error creating depth visualization: {e}")
        plt.close()
        return False


def visualize_segmentation_map(seg_img, output_path, scene_id, prefix="segmentation"):
    """
    Visualize segmentation map and save as image.
    
    Parameters:
    - seg_img: 2D numpy array containing segmentation IDs
    - output_path: Path to save the visualization
    - scene_id: Scene identifier
    - prefix: Prefix for the filename
    """
    try:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(20, 12))
        
        # Get unique segment IDs
        unique_ids = np.unique(seg_img)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
        
        # Create custom colormap for segmentation
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ids) + 1))
        colors[0] = [0, 0, 0, 1]  # Black for background
        custom_cmap = ListedColormap(colors)
        
        # Main segmentation visualization
        plt.subplot(2, 4, 1)
        plt.imshow(seg_img, cmap=custom_cmap, vmin=0, vmax=len(unique_ids))
        plt.colorbar(label='Segment ID')
        plt.title(f'Segmentation Map - {scene_id}')
        plt.axis('off')
        
        # Segmentation with different colormap
        plt.subplot(2, 4, 2)
        plt.imshow(seg_img, cmap='tab20')
        plt.colorbar(label='Segment ID')
        plt.title('Segmentation (Tab20 Colormap)')
        plt.axis('off')
        
        # Binary masks for each object
        plt.subplot(2, 4, 3)
        if len(unique_ids) > 0:
            # Show mask for the first non-background object
            target_id = unique_ids[0]
            target_mask = (seg_img == target_id).astype(np.uint8)
            plt.imshow(target_mask, cmap='gray')
            plt.title(f'Target Object {target_id} Mask')
            plt.axis('off')
        
        # Object count histogram
        plt.subplot(2, 4, 4)
        if len(unique_ids) > 0:
            object_counts = []
            for obj_id in unique_ids:
                count = np.sum(seg_img == obj_id)
                object_counts.append(count)
            
            plt.bar(unique_ids, object_counts, alpha=0.7)
            plt.xlabel('Object ID')
            plt.ylabel('Pixel Count')
            plt.title('Object Pixel Distribution')
        
        # Edge detection on segmentation
        plt.subplot(2, 4, 5)
        from scipy import ndimage
        # Create edge map
        edges = ndimage.sobel(seg_img.astype(np.float32))
        plt.imshow(edges, cmap='gray')
        plt.title('Segmentation Edges')
        plt.axis('off')
        
        # Combined visualization with outlines
        plt.subplot(2, 4, 6)
        # Create RGB image from segmentation
        seg_rgb = plt.cm.Set3(seg_img / (len(unique_ids) + 1))[:, :, :3]
        # Add edge outlines
        edge_mask = edges > 0.1
        seg_rgb[edge_mask] = [1, 1, 1]  # White edges
        plt.imshow(seg_rgb)
        plt.title('Segmentation with Outlines')
        plt.axis('off')
        
        # Segmentation statistics
        plt.subplot(2, 4, 7)
        stats_text = f"""
        Segmentation Statistics:
        Total objects: {len(unique_ids)}
        Object IDs: {unique_ids.tolist()}
        Image size: {seg_img.shape}
        Background pixels: {np.sum(seg_img == 0)}
        """
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        plt.axis('off')
        plt.title('Segmentation Statistics')
        
        # Individual object masks
        plt.subplot(2, 4, 8)
        if len(unique_ids) > 0:
            # Show all object masks in different colors
            all_masks = np.zeros_like(seg_img)
            for i, obj_id in enumerate(unique_ids):
                mask = (seg_img == obj_id).astype(np.uint8)
                all_masks += mask * (i + 1)
            
            plt.imshow(all_masks, cmap='tab10')
            plt.colorbar(label='Object Index')
            plt.title('All Object Masks')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        filename = f"{prefix}_{scene_id}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved segmentation visualization: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error creating segmentation visualization: {e}")
        plt.close()
        return False


def load_scene_data(data_dir, scene_id):
    """
    Load depth and segmentation data for a specific scene.
    
    Parameters:
    - data_dir: Directory containing the raw data
    - scene_id: Scene identifier
    
    Returns:
    - depth_img: Depth image array
    - seg_img: Segmentation image array
    """
    try:
        data_path = Path(data_dir)
        
        # Load depth data
        depth_path = data_path / f"depth_{scene_id}.npy"
        if depth_path.exists():
            depth_img = np.load(depth_path)
        else:
            print(f"Depth file not found: {depth_path}")
            return None, None
        
        # Load segmentation data
        seg_path = data_path / f"segmentation_{scene_id}.npy"
        if seg_path.exists():
            seg_img = np.load(seg_path)
        else:
            print(f"Segmentation file not found: {seg_path}")
            return None, None
        
        return depth_img, seg_img
        
    except Exception as e:
        print(f"Error loading scene data: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Visualize depth maps and segmentation maps from generated scenes")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing raw data files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for visualizations")
    parser.add_argument("--scene-id", type=str, help="Specific scene ID to visualize (optional)")
    parser.add_argument("--vis-depth", action="store_true", default=True, help="Visualize depth maps")
    parser.add_argument("--vis-segmentation", action="store_true", default=True, help="Visualize segmentation maps")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.scene_id:
        # Visualize specific scene
        depth_img, seg_img = load_scene_data(args.data_dir, args.scene_id)
        if depth_img is not None and seg_img is not None:
            if args.vis_depth:
                visualize_depth_map(depth_img, args.output_dir, args.scene_id)
            if args.vis_segmentation:
                visualize_segmentation_map(seg_img, args.output_dir, args.scene_id)
        else:
            print(f"Could not load data for scene {args.scene_id}")
    else:
        # Find all available scene data files
        depth_files = glob.glob(str(args.data_dir / "depth_*.npy"))
        seg_files = glob.glob(str(args.data_dir / "segmentation_*.npy"))
        
        # Extract scene IDs from depth files
        scene_ids = []
        for depth_file in depth_files:
            scene_id = Path(depth_file).stem.replace("depth_", "")
            scene_ids.append(scene_id)
        
        print(f"Found {len(scene_ids)} scenes to visualize")
        
        # Visualize each scene
        for i, scene_id in enumerate(scene_ids):
            print(f"Visualizing scene {i+1}/{len(scene_ids)}: {scene_id}")
            
            depth_img, seg_img = load_scene_data(args.data_dir, scene_id)
            if depth_img is not None and seg_img is not None:
                if args.vis_depth:
                    visualize_depth_map(depth_img, args.output_dir, scene_id)
                if args.vis_segmentation:
                    visualize_segmentation_map(seg_img, args.output_dir, scene_id)
            else:
                print(f"Could not load data for scene {scene_id}")
    
    print("Visualization complete!")


if __name__ == "__main__":
    main() 