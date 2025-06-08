import argparse
import json
from pathlib import Path
import os
import glob
from datetime import datetime
import subprocess

import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import sys
# sys.path = [
#     '../src/FGCGraspNet',
# ] + sys.path

# Grasp planner modules 
from src.vgn.detection import VGN
from src.vgn.detection_implicit import VGNImplicit

# Only keep target_sample_offline
from src.vgn.experiments import target_sample_offline_acronym   

# Utility
# (We keep set_random_seed here if you plan to reuse it, but it's not strictly used now.)
from src.vgn.utils.misc import set_random_seed

def str2bool(v):
    """
    Convert string inputs like 'yes', 'true', etc. to boolean values.
    Raises an error for invalid inputs.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_and_write_args_to_result_path(args):
    """
    Create a result directory based on model type and other arguments.
    Then write the command-line arguments to a text file in that directory.
    """
    # Only keep certain model types
    if args.type == "targo" or args.type == "targo_full_targ" or args.type == "targo_hunyun2" or args.type == "targo_ptv3" or args.type == "ptv3_scene":
        model_name = f'{args.type}'
    elif args.type in ("giga", "giga_aff", "vgn", "giga_hr"):
        model_name = f'{args.type}'
    elif args.type == "FGC-GraspNet" or args.type == "AnyGrasp":
        model_name = f'{args.type}'
    else:
        print("Unsupported type.")
        return

    result_directory = f'{args.result_root}/{model_name}'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_filename = f"{timestamp}"
    result_file_path = os.path.join(result_directory, result_filename)
    args.result_path = result_file_path

    args_dict = vars(args)
    args_content = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())

    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    if args.vis:
        args.logdir = Path(result_file_path)

    result_initial_path = f'{result_file_path}/initial.txt'
    with open(result_initial_path, 'w') as result_file:
        result_file.write(args_content)

    print(f"Args saved to {result_initial_path}")
    return result_file_path


def find_and_assign_first_checkpoint(args):
    """
    If the user sets '--model' to '.', automatically pick the first .pt checkpoint
    from the corresponding model directory.
    """
    # Only keep certain model types
    if args.type == "targo" or args.type == "targo_ptv3" or args.type == "ptv3_scene":
        model_name = f'{args.type}'
    elif args.type in ("giga", "giga_aff", "vgn", "giga_hr"):
        model_name = f'{args.type}'
    else:
        print("Unsupported type.")
        return


def main(args):
    """
    Main entry point: creates the grasp planner, then runs target_sample_offline.
    """
    # Choose VGN or VGNImplicit depending on 'args.type'
    if args.type == 'vgn':
        grasp_planner = VGN(
            args.model,
            args.type,
            best=args.best,
            qual_th=args.qual_th,
            force_detection=args.force,
            out_th=args.out_th if hasattr(args, 'out_th') and args.out_th is not None else 0.5,
            visualize=args.vis,
            cd_iou_measure=True
        )
    elif args.type in ['giga', 'giga_aff', 'giga_hr', 'targo', 'targo_full_targ', 'targo_hunyun2', 'targo_ptv3', 'ptv3_scene', 'FGC-GraspNet', 'AnyGrasp']:
        grasp_planner = VGNImplicit(
            args.model,
            args.type,
            best=args.best,
            qual_th=args.qual_th,
            force_detection=args.force,
            out_th=args.out_th if hasattr(args, 'out_th') and args.out_th is not None else 0.5,
            select_top=False,
            visualize=args.vis,
            cd_iou_measure=True,
        )
    else:
        raise NotImplementedError(f"Model type '{args.type}' is not implemented.")
    result_path = create_and_write_args_to_result_path(args)
    args.result_path = result_path
    # Only keep the target_sample_offline evaluation
    occ_level_sr = target_sample_offline_acronym.run(
        grasp_plan_fn=grasp_planner,
        data_type='acronym',
        logdir=args.logdir,
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        sim_gui=args.sim_gui,
        result_path=args.result_path,
        add_noise=None,
        sideview=args.sideview,
        visualize=args.vis,
        type=args.type,
        test_root=args.test_root,
        occ_level_dict_path=args.occ_level_dict,
        hunyun2_path=args.hunyun2_path,
        model_type=args.type,
        video_recording=args.video_recording,
        target_file_path=args.target_file,
    )

    # Save the result to a JSON file
    with open('occ_level_sr.json', 'w') as f:
        json.dump(occ_level_sr, f, indent=2)
    
    # Run category analysis after the test is completed
    if os.path.exists(f"{args.result_path}/meta_evaluations.txt"):
        print("\n====== Running category analysis ======")
        analysis_cmd = [
            "python", 
            "targo_eval_results/stastics_analysis/analyze_category.py", 
            "--eval_file", f"{args.result_path}/meta_evaluations.txt",
            "--output_dir", args.result_path,
            "--data_type", "acronym",
        ]
        print(f"Executing: {' '.join(analysis_cmd)}")
        subprocess.run(analysis_cmd)
        print("====== Category analysis completed ======\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", default="targo_hunyun2",
                        choices=["giga", "vgn", "targo", "targo_full_targ", "targo_hunyun2", "targo_ptv3", "ptv3_scene", 
                                "FGC-GraspNet", "AnyGrasp"],
                        help="Network to be used: vgn | giga_hr | giga_aff | giga | targo | targo_full_targ | targo_hunyun2 | targo_ptv3 | ptv3_scene | FGC-GraspNet | AnyGrasp")
    parser.add_argument("--occlusion-level", type=str, choices=["no", "slight", "medium"], default="medium",
                        help="Occlusion level for the experiment: no, slight or medium.")
    parser.add_argument("--hunyun2_path", type=str, default=None,
                        help="Path to hunyun2 model.")
    parser.add_argument("--result_root", type=Path,
                        default=None)
    parser.add_argument("--result-path", default='', type=str,
                        help="If empty, a new result path will be created automatically.")
    parser.add_argument("--logdir", type=Path, default=None,
                        help="Directory for storing logs or intermediate results.")
    parser.add_argument("--description", type=str, default="",
                        help="Optional experiment description.")
    parser.add_argument("--test_root", type=str,
                        default=None)
    parser.add_argument("--model", type=Path, default='/usr/stud/dira/GraspInClutter/targo/checkpoints/targonet.pt')
    parser.add_argument("--out_th", type=float, default=0.5,
                        help="Output threshold for valid grasps.")
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/train")
    parser.add_argument("--occ_level_dict", type=str, default=None)
    parser.add_argument("--sim-gui", type=str2bool, default=False,
                        help="Whether to enable a simulation GUI.")
    parser.add_argument("--qual-th", type=float, default=0.9,
                        help="Quality threshold for valid grasps.")
    parser.add_argument("--best", action="store_true", default=True,
                        help="Use the best valid grasp if available.")
    parser.add_argument("--force", action="store_true", default=True,
                        help="Force selection of a grasp even if below threshold.")
    parser.add_argument("--sideview", action="store_true", default=True,
                        help="Capture the scene from one side rather than top-down.")
    parser.add_argument("--vis", action="store_true", default=False,
                        help="Whether to visualize and save the affordance map.")
    parser.add_argument("--video-recording", type=str2bool, default=False,
                        help="Whether to record videos of grasping attempts.")
    parser.add_argument("--target-file", type=str, default='/usr/stud/dira/GraspInClutter/targo/example_targets/acronym_target_list.txt',
                        help="Path to a .txt file containing target names to record. If provided, only videos of these targets will be recorded.")
    
    args = parser.parse_args()
    
    # Set paths based on occlusion level
    if args.occlusion_level == "medium":
        if args.hunyun2_path is None:
            args.hunyun2_path = '/usr/stud/dira/GraspInClutter/Gen3DSR/hunyuan_results/acronym/medium'
        if args.result_root is None:
            args.result_root = 'targo_eval_results/acronym/eval_results_full-medium-occlusion'  
        if args.logdir is None:
            args.logdir = Path('targo_eval_results/acronym/eval_results_full-medium-occlusion')
        if args.test_root is None:
            args.test_root = 'data_scenes/acronym/acronym-middle-occlusion-1000'
        if args.occ_level_dict is None:
            args.occ_level_dict = '/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym-middle-occlusion-1000/test_set/occlusion_level_dict.json'
    elif args.occlusion_level == "slight":
        if args.hunyun2_path is None:
            args.hunyun2_path = '/usr/stud/dira/GraspInClutter/Gen3DSR/hunyuan_results/acronym/slight'
        if args.result_root is None:
            args.result_root = 'targo_eval_results/acronym/eval_results_full-slight-occlusion'
        if args.logdir is None:
            args.logdir = Path('targo_eval_results/acronym/eval_results_full-slight-occlusion')
        if args.test_root is None:
            args.test_root = 'data_scenes/acronym/acronym-slight-occlusion-1000'
        if args.occ_level_dict is None:
            args.occ_level_dict = 'data_scenes/acronym/acronym-slight-occlusion-1000/test_set/occlusion_level_dict.json'
    elif args.occlusion_level == "no":
        if args.hunyun2_path is None:
            args.hunyun2_path = '/usr/stud/dira/GraspInClutter/Gen3DSR/hunyuan_results/acronym/no'
        if args.result_root is None:
            args.result_root = 'targo_eval_results/acronym/eval_results_full-no-occlusion'
        if args.logdir is None:
            args.logdir = Path('targo_eval_results/acronym/eval_results_full-no-occlusion')
        if args.test_root is None:
            args.test_root = 'data_scenes/acronym/acronym-no-occlusion-1000'
        if args.occ_level_dict is None:
            args.occ_level_dict = 'data_scenes/acronym/acronym-no-occlusion-1000/test_set/occlusion_level_dict.json'

    # Removed unnecessary parameters:
    # --num-objects, --num-view, --num-rounds, --add-noise, --silence

    if args.logdir is None:
        if args.occlusion_level == "medium":
            args.logdir = Path('targo_eval_results/acronym/eval_results_full-medium-occlusion')
        elif args.occlusion_level == "slight":
            args.logdir = Path('targo_eval_results/acronym/eval_results_full-slight-occlusion')
        else:
            args.logdir = Path('targo_eval_results/acronym/eval_results_full-no-occlusion')

    # Automatically create a result path if none is provided
    if str(args.result_path) == "":
        main(args)