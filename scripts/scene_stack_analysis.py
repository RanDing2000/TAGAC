#!/usr/bin/env python3
import os
import sys
import argparse
import json
import math
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv


def find_glb_files(roots: List[str]) -> List[str]:
    files: List[str] = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".glb"):
                    files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def split_connected_components(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    try:
        parts = mesh.split(only_watertight=False)
        return [p for p in parts if isinstance(p, trimesh.Trimesh)]
    except Exception:
        return [mesh]


def apply_scene_transforms(scene: trimesh.Scene) -> List[trimesh.Trimesh]:
    # More efficient approach: directly access geometry and apply transforms
    meshes: List[trimesh.Trimesh] = []
    
    # Get all geometry objects and their transforms
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
            
        # Find all nodes that reference this geometry
        nodes = [n for n, gname in scene.graph.nodes_geometry if gname == name]
        
        if not nodes:
            # If no nodes reference this geometry, use it as-is
            meshes.extend(split_connected_components(geom))
        else:
            # Apply transforms from each node
            for node in nodes:
                try:
                    transform = scene.graph.get(node)[0]
                    if transform is not None:
                        m = geom.copy()
                        m.apply_transform(transform)
                        meshes.extend(split_connected_components(m))
                    else:
                        # No transform, use original
                        meshes.extend(split_connected_components(geom))
                except Exception:
                    # Fallback: use geometry without transform
                    meshes.extend(split_connected_components(geom))
    
    return meshes


def load_meshes_from_glb(glb_path: str, min_z_extent_ratio: float = 0.001, min_volume_ratio: float = 1e-6, verbose: bool = False) -> List[trimesh.Trimesh]:
    scene = trimesh.load(glb_path, process=False)
    meshes = []
    # Get raw geometries (still in local coordinates)
    for name, geom in scene.geometry.items():
        # print(f"{name}: {type(geom)}, v={len(geom.vertices)}, f={len(geom.faces)}")
        meshes.append(geom)
    return meshes




def xy_aabb(mesh: trimesh.Trimesh) -> Tuple[float, float, float, float]:
    mn = mesh.bounds[0]
    mx = mesh.bounds[1]
    return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])


def rect_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-12
    return float(inter / union)


def build_support_graph(meshes: List[trimesh.Trimesh], xy_iou_thr: float = 0.08, z_gap_thr: float = 0.01) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    n = len(meshes)
    if n <= 1:
        return [], {"n": n}

    aabbs = [xy_aabb(m) for m in meshes]
    tops = [float(m.bounds[1][2]) for m in meshes]
    bottoms = [float(m.bounds[0][2]) for m in meshes]

    edges: List[Tuple[int, int]] = []  # (lower, upper)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            iou = rect_iou(aabbs[i], aabbs[j])
            if iou < xy_iou_thr:
                continue
            z_gap = bottoms[j] - tops[i]
            if 0.0 <= z_gap <= z_gap_thr:
                edges.append((i, j))

    return edges, {"n": n}


def count_stacks(edges: List[Tuple[int, int]], n_objects: int) -> Dict[str, Any]:
    """
    Count distinct stacks using connected components in the support graph.
    A stack is a connected component where objects support each other vertically.
    Only groups with 2 or more objects are considered as stacks.
    """
    if n_objects == 0:
        return {"n_stacks": 0, "n_objects": 0, "stack_sizes": [], "objects_per_stack": [], "individual_objects": 0}
    
    if not edges:
        # No support relations = no stacks, all objects are individual
        return {
            "n_stacks": 0,
            "n_objects": n_objects,
            "stack_sizes": [],
            "objects_per_stack": [],
            "individual_objects": n_objects
        }
    
    # Build adjacency list for undirected graph (support relation is bidirectional for grouping)
    adj = [[] for _ in range(n_objects)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Find connected components using DFS
    visited = [False] * n_objects
    stacks = []
    
    def dfs(node: int, component: List[int]) -> None:
        visited[node] = True
        component.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(n_objects):
        if not visited[i]:
            component: List[int] = []
            dfs(i, component)
            # Only consider components with 2 or more objects as stacks
            if len(component) >= 2:
                stacks.append(component)
    
    stack_sizes = [len(stack) for stack in stacks]
    individual_objects = n_objects - sum(stack_sizes)  # Objects not in any stack
    
    return {
        "n_stacks": len(stacks),
        "n_objects": n_objects,
        "stack_sizes": stack_sizes,
        "objects_per_stack": stack_sizes,  # Same as stack_sizes for compatibility
        "stack_details": stacks,  # List of lists, each inner list contains object indices in that stack
        "individual_objects": individual_objects  # Objects that are not part of any stack
    }


def analyze_scene(glb_path: str, xy_iou_thr: float, z_gap_thr: float, verbose: bool = False) -> Dict[str, Any]:
    meshes = load_meshes_from_glb(glb_path, verbose=verbose)
    n_obj = len(meshes)
    if n_obj > 9:
        print(f"n_obj: {n_obj} in {glb_path}")
    if n_obj == 0:
        return {
            "path": glb_path,
            "n_objects": 0,
            "n_stacks": 0,
            "individual_objects": 0,
            "area_xy": 0.0,
            "obj_density": 0.0,
            "stack_density": 0.0,
        }

    # Scene XY area from overall AABB
    mins = np.vstack([m.bounds[0] for m in meshes]).min(axis=0)
    maxs = np.vstack([m.bounds[1] for m in meshes]).max(axis=0)
    area_xy = float(max(0.0, (maxs[0] - mins[0])) * max(0.0, (maxs[1] - mins[1])))

    edges, _ = build_support_graph(meshes, xy_iou_thr=xy_iou_thr, z_gap_thr=z_gap_thr)
    stack_info = count_stacks(edges, n_obj)
    n_stacks = stack_info["n_stacks"]
    stack_sizes = stack_info["stack_sizes"]
    individual_objects = stack_info["individual_objects"]

    obj_density = (n_obj / area_xy) if area_xy > 0 else 0.0
    stack_density = (n_stacks / area_xy) if area_xy > 0 else 0.0
    
    # Additional statistics
    avg_stack_size = sum(stack_sizes) / len(stack_sizes) if stack_sizes else 0.0
    max_stack_size = max(stack_sizes) if stack_sizes else 0
    min_stack_size = min(stack_sizes) if stack_sizes else 0

    return {
        "path": glb_path,
        "n_objects": n_obj,
        "n_stacks": n_stacks,
        "individual_objects": individual_objects,
        "area_xy": area_xy,
        "obj_density": obj_density,
        "stack_density": stack_density,
        "stack_sizes": stack_sizes,
        "avg_stack_size": avg_stack_size,
        "max_stack_size": max_stack_size,
        "min_stack_size": min_stack_size,
    }


def save_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    if not rows:
        return
    keys = [
        "dataset",
        "path", 
        "n_objects",
        "n_stacks",
        "individual_objects",
        "area_xy",
        "obj_density",
        "stack_density",
        "avg_stack_size",
        "max_stack_size",
        "min_stack_size",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def plot_hist(values: List[float], title: str, out_path: str, bins: int = 30, xlabel: str = "") -> None:
    if len(values) == 0:
        return
    plt.figure(figsize=(10, 7))
    
    # Create histogram
    n, bins, patches = plt.hist(values, bins=bins, edgecolor="black", alpha=0.7)
    
    # Add value labels on top of each bar
    for i in range(len(n)):
        if n[i] > 0:  # Only label bars with values > 0
            # Calculate bar center position
            bar_center = (bins[i] + bins[i+1]) / 2
            # Add text label above the bar
            plt.text(bar_center, n[i], str(int(n[i])), 
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add statistics text on the plot
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    total_count = len(values)
    
    # Format statistics text
    stats_text = f'Total: {total_count}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
    
    # Add text box with statistics
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to make room for labels
    plt.ylim(0, max(n) * 1.1)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze N_obj and N_stacks over GLB scenes")
    parser.add_argument("--roots", nargs="+", default=["datasets/messy_kitchen_scenes_part1", "datasets/messy_kitchen_scenes_part2"], help="Dataset root directories to traverse")
    parser.add_argument("--output_dir", type=str, default="results/stack_analysis", help="Output directory")
    parser.add_argument("--xy_iou_thr", type=float, default=0.08, help="XY IoU threshold [0.05~0.15]")
    parser.add_argument("--z_gap", type=float, default=0.01, help="Z gap threshold in scene units (e.g., meters)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    glb_files = find_glb_files(args.roots)
    print(f"Found {len(glb_files)} GLB files")

    rows: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0
    
    for i, p in enumerate(glb_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(glb_files)} ({i/len(glb_files)*100:.1f}%) - Success: {success_count}, Failed: {fail_count}")
        
        dataset = ""
        for r in args.roots:
            if p.startswith(os.path.abspath(r)):
                dataset = os.path.basename(r.rstrip(os.sep))
                break
        try:
            res = analyze_scene(p, xy_iou_thr=args.xy_iou_thr, z_gap_thr=args.z_gap, verbose=args.verbose)
            res["dataset"] = dataset
            rows.append(res)
            stack_sizes_str = str(res['stack_sizes']) if len(res['stack_sizes']) <= 10 else f"{res['stack_sizes'][:5]}...+{len(res['stack_sizes'])-5}more"
            print(f"OK: {p} -> objects={res['n_objects']}, stacks={res['n_stacks']}, individual={res['individual_objects']}, stack_sizes={stack_sizes_str}")
            success_count += 1
        except Exception as e:
            print(f"FAIL: {p} due to {e}")
            import traceback
            if args.verbose:
                print(f"  Full error traceback:")
                traceback.print_exc()
            # Add a failed entry to continue processing
            failed_res = {
                "dataset": dataset,
                "path": p,
                "n_objects": -1,
                "n_stacks": -1,
                "individual_objects": -1,
                "area_xy": -1.0,
                "obj_density": -1.0,
                "stack_density": -1.0,
                "stack_sizes": [],
                "avg_stack_size": -1.0,
                "max_stack_size": -1,
                "min_stack_size": -1,
                "error": str(e)
            }
            rows.append(failed_res)
            fail_count += 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.output_dir, f"stack_stats_{timestamp}.csv")
    save_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")
    print(f"Final Summary: Success: {success_count}, Failed: {fail_count}, Total: {len(glb_files)}")

    # Aggregate and plot (only for successful cases)
    successful_rows = [r for r in rows if r.get("n_objects", -1) >= 0]
    print(f"Generating plots for {len(successful_rows)} successful cases...")
    
    if len(successful_rows) > 0:
        n_objs = [int(r["n_objects"]) for r in successful_rows]
        n_stacks = [int(r["n_stacks"]) for r in successful_rows]
        individual_objs = [int(r["individual_objects"]) for r in successful_rows]
        obj_density = [float(r["obj_density"]) for r in successful_rows]
        stack_density = [float(r["stack_density"]) for r in successful_rows]
        avg_stack_sizes = [float(r["avg_stack_size"]) for r in successful_rows]
        max_stack_sizes = [int(r["max_stack_size"]) for r in successful_rows]
        
        # Flatten all stack sizes for distribution
        all_stack_sizes = []
        for r in successful_rows:
            all_stack_sizes.extend(r["stack_sizes"])

        # Print summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total scenes analyzed: {len(successful_rows)}")
        print(f"Total objects across all scenes: {sum(n_objs)}")
        print(f"Total stacks across all scenes: {sum(n_stacks)}")
        print(f"Total individual objects: {sum(individual_objs)}")
        print(f"Average objects per scene: {np.mean(n_objs):.2f} ± {np.std(n_objs):.2f}")
        print(f"Average stacks per scene: {np.mean(n_stacks):.2f} ± {np.std(n_stacks):.2f}")
        print(f"Average individual objects per scene: {np.mean(individual_objs):.2f} ± {np.std(individual_objs):.2f}")
        
        if all_stack_sizes:
            print(f"Total objects in stacks: {sum(all_stack_sizes)}")
            print(f"Average stack size: {np.mean(all_stack_sizes):.2f} ± {np.std(all_stack_sizes):.2f}")
            print(f"Largest stack found: {max(all_stack_sizes)} objects")
            print(f"Most common stack sizes: {sorted(set(all_stack_sizes), key=lambda x: all_stack_sizes.count(x), reverse=True)[:5]}")
        
        print(f"Average object density: {np.mean(obj_density):.4f} ± {np.std(obj_density):.4f} objects/m²")
        print(f"Average stack density: {np.mean(stack_density):.4f} ± {np.std(stack_density):.4f} stacks/m²")
        print(f"========================\n")

        # Generate plots with enhanced titles
        plot_hist(n_objs, f"Histogram of N_objects (Total: {len(successful_rows)} scenes)", 
                 os.path.join(args.output_dir, f"hist_n_objects_{timestamp}.png"), xlabel="N_objects")
        plot_hist(n_stacks, f"Histogram of N_stacks (Total: {len(successful_rows)} scenes)", 
                 os.path.join(args.output_dir, f"hist_n_stacks_{timestamp}.png"), xlabel="N_stacks")
        plot_hist(individual_objs, f"Histogram of N_individual_objects (Total: {len(successful_rows)} scenes)", 
                 os.path.join(args.output_dir, f"hist_n_individual_objects_{timestamp}.png"), xlabel="N_individual_objects")
        plot_hist(obj_density, f"Histogram of object density (Total: {len(successful_rows)} scenes)", 
                 os.path.join(args.output_dir, f"hist_obj_density_{timestamp}.png"), xlabel="objects / m²")
        plot_hist(stack_density, f"Histogram of stack density (Total: {len(successful_rows)} scenes)", 
                 os.path.join(args.output_dir, f"hist_stack_density_{timestamp}.png"), xlabel="stacks / m²")
        plot_hist(avg_stack_sizes, f"Histogram of average stack size (Total: {len(successful_rows)} scenes)", 
                 os.path.join(args.output_dir, f"hist_avg_stack_size_{timestamp}.png"), xlabel="avg objects per stack")
        plot_hist(max_stack_sizes, f"Histogram of max stack size (Total: {len(successful_rows)} scenes)", 
                 os.path.join(args.output_dir, f"hist_max_stack_size_{timestamp}.png"), xlabel="max objects per stack")
        plot_hist(all_stack_sizes, f"Histogram of all stack sizes (Total: {sum(all_stack_sizes)} objects in stacks)", 
                 os.path.join(args.output_dir, f"hist_all_stack_sizes_{timestamp}.png"), xlabel="objects per stack")
        
        print(f"Generated {8} histogram plots in {args.output_dir}")
    else:
        print("No successful cases to plot.")

    print("Done.")


if __name__ == "__main__":
    main()


