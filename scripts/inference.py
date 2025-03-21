import argparse
import json
from pathlib import Path
import os
import glob
from datetime import datetime

import numpy as np

# Grasp planner modules
from src.vgn.detection import VGN
from src.vgn.detection_implicit import VGNImplicit

# Only keep target_sample_offline
from src.vgn.experiments import target_sample_offline_2

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
    if args.type == "targo":
        model_name = f'{args.type}'
    elif args.type in ("giga", "giga_aff", "vgn", "giga_hr"):
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


def find_and_assign_first_checkpoint(args):
    """
    If the user sets '--model' to '.', automatically pick the first .pt checkpoint
    from the corresponding model directory.
    """
    # Only keep certain model types
    if args.type == "targo":
        model_name = f'{args.type}'
    elif args.type in ("giga", "giga_aff", "vgn", "giga_hr"):
        model_name = f'{args.type}'
    else:
        print("Unsupported type.")
        return

    # model_directory = os.path.join(args.model_root, model_name)
    # checkpoint_pattern = os.path.join(model_directory, '*.pt')
    # checkpoint_files = glob.glob(checkpoint_pattern)

    # if len(checkpoint_files) > 0:
    #     args.model = checkpoint_files[0]
    #     print("Assigned model:", args.model)
    # else:
    #     print("No checkpoint files found.")


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
            out_th=0.1,  # You can adjust this threshold as needed
            visualize=args.vis
        )
    elif args.type in ['giga', 'giga_aff', 'giga_hr', 'targo']:
        grasp_planner = VGNImplicit(
            args.model,
            args.type,
            best=args.best,
            qual_th=args.qual_th,
            force_detection=args.force,
            out_th=0.15,  # You can adjust this threshold as needed
            select_top=False,
            visualize=args.vis,
        )
    else:
        raise NotImplementedError(f"Model type '{args.type}' is not implemented.")

    # Only keep the target_sample_offline evaluation
    occ_level_sr = target_sample_offline_2.run(
        grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        sim_gui=args.sim_gui,
        result_path=args.result_path,
        add_noise=args.add_noise,
        sideview=args.sideview,
        visualize=args.vis,
        type=args.type,
        test_root=args.test_root,
    )

    # Save the result to a JSON file
    with open('occ_level_sr.json', 'w') as f:
        json.dump(occ_level_sr, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Keep only the necessary arguments
    parser.add_argument("--type", default="targo",
                        choices=["giga_hr", "giga_aff", "giga", "vgn", "targo"],
                        help="Model type: giga_hr | giga_aff | giga | vgn | targo")
    parser.add_argument("--result_root", type=Path,
                        default='eval_results_test/')
    parser.add_argument("--result-path", default='', type=str,
                        help="If empty, a new result path will be created automatically.")
    parser.add_argument("--logdir", type=Path, default='eval_results/',
                        help="Directory for storing logs or intermediate results.")
    # parser.add_argument("--logdir", type=Path, default='eval_results_train/',
    #                     help="Directory for storing logs or intermediate results.")
    parser.add_argument("--description", type=str, default="",
                        help="Optional experiment description.")
    parser.add_argument("--test_root", type=str,
                        default='/storage/user/dira/nips_data_version6/test_set_gaussian_0.002')
                        # default='/storage/user/dira/nips_data_version6/combined/test_set_gaussian_0.002_train_combined/combined')
    parser.add_argument("--model", type=Path, default='checkpoints/targonet.pt')
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/test")
    # parser.add_argument("--object-set", type=str, default="packed/train")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--sim-gui", type=str2bool, default=False,
                        help="Whether to enable a simulation GUI.")
    parser.add_argument("--qual-th", type=float, default=0.9,
                        help="Quality threshold for valid grasps.")
    parser.add_argument("--best", action="store_true", default=True,
                        help="Use the best valid grasp if available.")
    parser.add_argument("--force", action="store_true", default=True,
                        help="Force selection of a grasp even if below threshold.")
    parser.add_argument("--add-noise", type=str, default="dex",
                        help="Type of noise added to depth or input data: 'trans', 'dex', 'norm' or none.")
    parser.add_argument("--sideview", action="store_true", default=True,
                        help="Capture the scene from one side rather than top-down.")
    parser.add_argument("--silence", action="store_true",
                        help="Disable the tqdm progress bar.")
    parser.add_argument("--vis", action="store_true", default=True,
                        help="Whether to visualize and save the affordance map.")
    # parser.add_argument("--vis", action="store_true", default=False,
    #                     help="Whether to visualize and save the affordance map.")

    args = parser.parse_args()

    # Automatically find the first checkpoint if model is '.'
    if str(args.model) == ".":
        find_and_assign_first_checkpoint(args)

    # Automatically create a result path if none is provided
    if str(args.result_path) == "":
        create_and_write_args_to_result_path(args)
    main(args)
