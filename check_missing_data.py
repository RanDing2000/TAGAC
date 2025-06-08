#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def check_missing_data(dataset_root="/usr/stud/dira/GraspInClutter/targo/data_scenes/targo_dataset"):
    """
    检查targo数据集中缺失的数据:
    1. single scene (_s_) 文件中缺失的 complete_target_tsdf
    2. 所有scene文件中缺失的 pc_depth_targ 数据
    """
    dataset_root = Path(dataset_root)
    scenes_dir = dataset_root / "scenes"
    
    print("=" * 80)
    print("TARGO Dataset Missing Data Check")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Scenes directory: {scenes_dir}")
    
    if not scenes_dir.exists():
        print(f"ERROR: Scenes directory not found: {scenes_dir}")
        return
    
    # 获取所有scene文件
    all_scene_files = list(scenes_dir.glob("*.npz"))
    total_files = len(all_scene_files)
    print(f"Total scene files found: {total_files}\n")
    
    # 分类scene文件
    single_scenes = [f for f in all_scene_files if "_s_" in f.stem]
    clutter_scenes = [f for f in all_scene_files if "_c_" in f.stem]
    double_scenes = [f for f in all_scene_files if "_d_" in f.stem]
    print(f"Single scenes (_s_): {len(single_scenes)}")
    print(f"Clutter scenes (_c_): {len(clutter_scenes)}")
    print(f"Double scenes (_d_): {len(double_scenes)}\n")
    
    # 准备统计容器
    missing_complete_target = []
    missing_targ_pc = defaultdict(list)
    scenes_with_targ_pc = defaultdict(list)
    type_map = {"_s_": "single", "_c_": "clutter", "_d_": "double"}
    
    # 单轮遍历，检查两项
    for scene_file in tqdm(all_scene_files, desc="Scanning all scenes"):
        sid = scene_file.stem
        # 场景类型判断
        stype = "unknown"
        for marker, name in type_map.items():
            if marker in sid:
                stype = name
                break
        
        try:
            with np.load(scene_file, allow_pickle=True) as data:
                # 1) single 场景缺失 complete_target_tsdf
                if stype == "single":
                    if "complete_target_tsdf" not in data:
                        missing_complete_target.append(sid)
                
                # 2) 全部场景缺失 pc_depth_targ
                if "pc_depth_targ" in data:
                    scenes_with_targ_pc[stype].append(sid)
                else:
                    missing_targ_pc[stype].append(sid)
        except Exception as e:
            # 任何读盘错误都算缺失
            if stype == "single":
                missing_complete_target.append(sid)
            missing_targ_pc[stype].append(sid)
    
    # 输出第一部分结果
    print("\n" + "=" * 60)
    print("1. Missing 'complete_target_tsdf' in single scenes")
    print("=" * 60)
    print(f"Total single scenes: {len(single_scenes)}")
    print(f"  Missing complete_target_tsdf: {len(missing_complete_target)} "
          f"({len(missing_complete_target)/len(single_scenes)*100:.1f}%)")
    if missing_complete_target:
        print("  Examples:")
        for sid in missing_complete_target[:20]:
            print(f"    {sid}")
        if len(missing_complete_target) > 20:
            print(f"    ... and {len(missing_complete_target)-20} more")
    
    # 输出第二部分结果
    print("\n" + "=" * 60)
    print("2. Missing 'pc_depth_targ' in all scenes")
    print("=" * 60)
    total_missing_targ = sum(len(v) for v in missing_targ_pc.values())
    print(f"Total scenes missing pc_depth_targ: {total_missing_targ} "
          f"({total_missing_targ/total_files*100:.1f}%)\n")
    for stype in ["single", "clutter", "double", "unknown"]:
        has = len(scenes_with_targ_pc[stype])
        miss = len(missing_targ_pc[stype])
        tot = has + miss
        if tot == 0:
            continue
        print(f"{stype.capitalize()} scenes: total={tot}, with_pc={has}, missing_pc={miss}")
        if miss:
            for sid in missing_targ_pc[stype][:10]:
                print(f"    {sid}")
            if miss > 10:
                print(f"    ... and {miss-10} more")
    
    # 检查已有日志
    print("\n" + "=" * 60)
    print("3. Existing miss log file")
    print("=" * 60)
    miss_log = dataset_root.parent / "miss_complete_target_tsdf.txt"
    if miss_log.exists():
        logged = [l.strip() for l in open(miss_log) if l.strip()]
        print(f"Previously logged missing: {len(logged)}")
        for sid in logged[:10]:
            print(f"    {sid}")
        if len(logged) > 10:
            print(f"    ... and {len(logged)-10} more")
    else:
        print(f"No log file at: {miss_log}")
    
    # 样本文件键分析
    print("\n" + "=" * 60)
    print("4. Sample data keys analysis (first 5 files)")
    print("=" * 60)
    for scene_file in all_scene_files[:5]:
        sid = scene_file.stem
        try:
            with np.load(scene_file, allow_pickle=True) as data:
                keys = sorted(data.keys())
                print(f"{sid}: keys = {keys}")
                for key in ["pc_depth_targ", "pc_depth_scene", "complete_target_tsdf", "grid_scene", "grid_targ"]:
                    if key in data:
                        val = data[key]
                        print(f"  {key}: shape={getattr(val, 'shape', type(val))}")
                    else:
                        print(f"  {key}: MISSING")
        except Exception as e:
            print(f"ERROR reading {sid}: {e}")
    
    # 保存报告
    outdir = Path("data_check_results")
    outdir.mkdir(exist_ok=True)
    with open(outdir/"missing_complete_target_tsdf.txt", "w") as f:
        f.write("# Single scenes missing complete_target_tsdf\n")
        for sid in missing_complete_target:
            f.write(f"{sid}\n")
    with open(outdir/"missing_targ_pc.txt", "w") as f:
        f.write("# Scenes missing pc_depth_targ\n")
        for stype, lst in missing_targ_pc.items():
            if not lst: continue
            f.write(f"\n# {stype.upper()} ({len(lst)})\n")
            for sid in lst:
                f.write(f"{sid}\n")
    
    print(f"\nResults saved under '{outdir}/'")

if __name__ == "__main__":
    check_missing_data()
