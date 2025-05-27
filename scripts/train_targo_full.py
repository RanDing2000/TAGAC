#!/usr/bin/env python3
"""
Training script for TARGO Full - uses complete target point clouds without shape completion.

This script trains an original TARGO model using preprocessed complete target point clouds
instead of using AdaPoinTr for shape completion. The network structure is the original TARGO
but uses complete target meshes as ground truth.
"""

import argparse
from pathlib import Path
import numpy as np
np.int = int
np.bool = bool
from datetime import datetime
from torch.utils.data import DataLoader, Subset, random_split
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy, Precision, Recall
import torch
import open3d as o3d
import time
from torch.utils import tensorboard
import torch.nn.functional as F
from vgn.dataset_voxel import DatasetVoxel_Target
from vgn.networks import get_network, load_network
from vgn.utils.transform import Transform
from vgn.perception import TSDFVolume
from utils_giga import visualize_and_save_tsdf, save_point_cloud_as_ply, tsdf_to_ply, points_to_voxel_grid_batch, pointcloud_to_voxel_indices
from utils_giga import filter_and_pad_point_clouds
import numpy as np
from vgn.utils.transform import Rotation, Transform
from src.vgn.io import read_complete_target_pc, check_complete_target_available

LOSS_KEYS = ['loss_all', 'loss_qual', 'loss_rot', 'loss_width']

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    if args.savedir == '':
        # create log directory
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "{}_dataset={},augment={},net={},batch_size={},lr={:.0e},{}".format(
            time_stamp,
            args.dataset.name,
            args.augment,
            args.net,
            args.batch_size,
            args.lr,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.dataset_raw, args.batch_size, args.val_split, args.augment, 
        args.complete_shape, args.targ_grasp, args.set_theory, args.ablation_dataset,
        args.data_contain, args.decouple, args.use_complete_targ, args.net, 
        args.input_points, args.shape_completion, args.vis_data, args.logdir, kwargs
    )

    # build the network or load
    if args.load_path == '':
        net = get_network(args.net).to(device)
    else:
        net = load_network(args.load_path, device, args.net)

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # define metrics
    metrics = {
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
        "precision": Precision(lambda out: (torch.round(out[1][0]), out[2][0])),
        "recall": Recall(lambda out: (torch.round(out[1][0]), out[2][0])),
    }
    
    for k in LOSS_KEYS:
        metrics[k] = Average(lambda out, sk=k: out[3][sk])

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device, args.input_points, args.net)
    evaluator = create_evaluator(net, loss_fn, metrics, device, args.input_points, args.net)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True, dynamic_ncols=True, disable=args.silence).attach(trainer)

    train_writer, val_writer = create_summary_writers(net, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_learning_rate(engine):
        current_lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar("learning_rate", current_lr, engine.state.epoch)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def update_scheduler(engine):
        if engine.state.epoch % args.lr_schedule_interval == 0:
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        for k, v in metrics.items():
            train_writer.add_scalar(k, v, epoch)

        msg = 'Train'
        for k, v in metrics.items():
            msg += f' {k}: {v:.4f}'
        print(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        for k, v in metrics.items():
            val_writer.add_scalar(k, v, epoch)
            
        msg = 'Val'
        for k, v in metrics.items():
            msg += f' {k}: {v:.4f}'
        print(msg)

    def default_score_fn(engine):
        score = engine.state.metrics['accuracy']
        return score

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "vgn",
        n_saved=100,
        require_empty=True,
    )
    best_checkpoint_handler = ModelCheckpoint(
        logdir,
        "best_vgn",
        n_saved=100,
        score_name="val_acc",
        score_function=default_score_fn,
        require_empty=True,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, best_checkpoint_handler, {args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)

def create_train_val_loaders(root, root_raw, batch_size, val_split, augment, complete_shape, targ_grasp, set_theory, ablation_dataset, data_contain, decouple, use_complete_targ, model_type, input_points, shape_completion, vis_data, logdir, kwargs):
    # Load the dataset with complete target configuration
    dataset = DatasetVoxelTargetFull(root, root_raw, augment=augment, ablation_dataset=ablation_dataset, model_type=model_type,
                                    data_contain=data_contain, decouple=decouple, use_complete_targ=use_complete_targ, 
                                    input_points=input_points, shape_completion=shape_completion, vis_data=vis_data, logdir=logdir)
    
    scene_ids = dataset.df['scene_id'].tolist()

    # Extract identifiers for clutter scenes
    clutter_ids = set(scene_id for scene_id in scene_ids if '_c_' in scene_id)
    
    # Randomly sample 10% of clutter scenes for validation
    val_clutter_ids_set = set(np.random.choice(list(clutter_ids), size=int(val_split * len(clutter_ids)), replace=False))

    # Extract base identifiers for single and double scenes that should not be in the training set
    related_single_double_ids = {id.replace('_c_', '_s_') for id in val_clutter_ids_set} | \
                                {id.replace('_c_', '_d_') for id in val_clutter_ids_set}

    # Create train and validation indices
    val_indices = [i for i, id in enumerate(scene_ids) if id in val_clutter_ids_set]
    train_indices = [i for i, id in enumerate(scene_ids) if id not in val_clutter_ids_set and id not in related_single_double_ids]

    # Create subsets for training and validation
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    
    return train_loader, val_loader

class DatasetVoxelTargetFull(DatasetVoxel_Target):
    """
    Extended dataset class for TARGO Full that loads complete target point clouds
    from preprocessed data instead of using shape completion.
    """
    
    def __init__(self, root, raw_root, num_point=2048, augment=False, ablation_dataset="", model_type="targo",
                 data_contain="pc and targ_grid", add_single_supervision=False, decouple=False, use_complete_targ=True,
                 input_points='tsdf_points', shape_completion=False, vis_data=False, logdir=None):
        
        # Force use_complete_targ=True and shape_completion=False for TARGO Full
        super().__init__(root, raw_root, num_point, augment, ablation_dataset, model_type,
                        data_contain, add_single_supervision, decouple, use_complete_targ=True,
                        input_points=input_points, shape_completion=False, vis_data=vis_data, logdir=logdir)
        
        print("=" * 60)
        print("TARGO Full Dataset Configuration:")
        print("=" * 60)
        print(f"Using complete target point clouds: {self.use_complete_targ}")
        print(f"Shape completion disabled: {not self.shape_completion}")
        print(f"Model type: {self.model_type}")
        print(f"Data contain: {self.data_contain}")
        print("=" * 60)
        
        # Verify that complete target data is available
        self._verify_complete_target_availability()
    
    def _verify_complete_target_availability(self):
        """Verify that complete target data is available for the dataset."""
        print("Verifying complete target data availability...")
        
        # Check a sample of scenes
        sample_scenes = self.df['scene_id'].sample(min(10, len(self.df))).tolist()
        available_count = 0
        
        for scene_id in sample_scenes:
            if check_complete_target_available(self.root, scene_id):
                available_count += 1
        
        if available_count == 0:
            raise RuntimeError(
                "No complete target data found in dataset. "
                "Please run the preprocessing script first:\n"
                "python scripts/preprocess_complete_target_mesh.py "
                f"--raw_root {self.raw_root} --output_root {self.root}"
            )
        elif available_count < len(sample_scenes):
            print(f"Warning: Only {available_count}/{len(sample_scenes)} sample scenes have complete target data")
        else:
            print(f"✓ Complete target data verified for {available_count}/{len(sample_scenes)} sample scenes")
    
    def __getitem__(self, i):
        """Override to use complete target point clouds directly."""
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        
        if not self.model_type == "vgn":
            pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
            width = np.float32(self.df.loc[i, "width"])
            label = self.df.loc[i, "label"].astype(np.int64)
        else:
            pos = self.df.loc[i, "i":"k"].to_numpy(np.single)
            width = self.df.loc[i, "width"].astype(np.single)
            label = self.df.loc[i, "label"].astype(np.int64)

        if self.data_contain == "pc and targ_grid":
            # Load voxel grids
            from src.vgn.io import read_voxel_and_mask_occluder
            voxel_grid, targ_grid = read_voxel_and_mask_occluder(self.raw_root, scene_id)
            
            # Load complete target point cloud from preprocessed data
            complete_target_pc = read_complete_target_pc(self.root, scene_id)
            if complete_target_pc is None:
                # Fallback to original target point cloud if complete target not available
                from src.vgn.io import read_targ_pc
                from utils_giga import points_within_boundary, specify_num_points
                complete_target_pc = read_targ_pc(self.raw_root, scene_id).astype(np.float32)
                complete_target_pc = points_within_boundary(complete_target_pc)
                print(f"Warning: Using fallback target PC for {scene_id}")
            
            # Load scene point cloud (without target)
            from src.vgn.io import read_scene_no_targ_pc
            from utils_giga import points_within_boundary, specify_num_points
            scene_no_targ_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
            scene_no_targ_pc = points_within_boundary(scene_no_targ_pc)
            
            # Combine scene and complete target point clouds
            scene_pc = np.concatenate((scene_no_targ_pc, complete_target_pc), axis=0)
            
            # Normalize point counts
            targ_pc = specify_num_points(complete_target_pc, 2048)
            scene_pc = specify_num_points(scene_pc, 2048)
            
            # Add plane points
            plane = np.load('/usr/stud/dira/GraspInClutter/grasping/setup/plane_sampled.npy')
            scene_pc = np.concatenate((scene_pc, plane), axis=0)
            
            # Normalize coordinates
            targ_pc = targ_pc / 0.3 - 0.5
            scene_pc = scene_pc / 0.3 - 0.5
            
            if self.vis_data:
                vis_path = str(self.vis_logdir / f'complete_targ_pc_{i}.ply')
                save_point_cloud_as_ply(targ_pc, vis_path)
                vis_path = str(self.vis_logdir / f'scene_pc_{i}.ply')
                save_point_cloud_as_ply(scene_pc, vis_path)

            x = (voxel_grid[0], targ_grid[0], targ_pc, scene_pc)

        elif self.data_contain == "pc":
            from src.vgn.io import read_voxel_and_mask_occluder
            voxel_grid, _ = read_voxel_and_mask_occluder(self.root, scene_id)
            x = (voxel_grid[0])

        # Apply data augmentation if enabled
        if self.augment:
            from src.vgn.dataset_voxel import apply_transform
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)

        # Normalize position and width
        if self.model_type != "vgn":
            pos = pos / self.size - 0.5
            width = width / self.size

            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()
        else:
            index = np.round(pos).astype(np.int64)
            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()

        y = (label, rotations, width)
        
        if self.model_type == "vgn":
            return x, y, index
        else:
            return x, y, pos

def prepare_batch(batch, device, model_type="targo"):
    """Prepare batch data for TARGO Full model."""
    (pc, targ_grid, targ_pc, scene_pc), (label, rotations, width), pos = batch

    # Convert to device and proper types
    pc = pc.float().to(device)
    targ_grid = targ_grid.float().to(device)
    targ_pc = targ_pc.float().to(device)
    scene_pc = scene_pc.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    width = width.float().to(device)
    pos.unsqueeze_(1)  # B, 1, 3
    pos = pos.float().to(device)

    # For TARGO Full, return scene and complete target point clouds
    return (scene_pc, targ_pc), (label, rotations, width), pos

def select(out):
    """Select outputs from model predictions."""
    qual_out, rot_out, width_out = out
    rot_out = rot_out.squeeze(1)
    return qual_out.squeeze(-1), rot_out, width_out.squeeze(-1)

def loss_fn(y_pred, y):
    """Loss function for TARGO Full."""
    label_pred, rotation_pred, width_pred = y_pred
    label, rotations, width = y
    
    loss_qual = _qual_loss_fn(label_pred, label)
    loss_rot = _rot_loss_fn(rotation_pred, rotations)
    loss_width = _width_loss_fn(width_pred, width)
    loss = loss_qual + label * (loss_rot + 0.01 * loss_width)
    
    loss_dict = {
        'loss_qual': loss_qual.mean(),
        'loss_rot': loss_rot.mean(),
        'loss_width': loss_width.mean(),
        'loss_all': loss.mean()
    }

    return loss.mean(), loss_dict

def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none")

def _rot_loss_fn(pred, target):
    loss0 = _quat_loss_fn(pred, target[:, 0])
    loss1 = _quat_loss_fn(pred, target[:, 1])
    return torch.min(loss0, loss1)

def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

def _width_loss_fn(pred, target):
    return F.mse_loss(40 * pred, 40 * target, reduction="none")

def create_trainer(net, optimizer, loss_fn, metrics, device, input_points, model_type="targo"):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()
        
        x, y, pos = prepare_batch(batch, device, model_type)
        y_pred = select(net(x, pos))
        loss, loss_dict = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss_dict

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def create_evaluator(net, loss_fn, metrics, device, input_points, model_type="targo"):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, pos = prepare_batch(batch, device, model_type)
            y_pred = select(net(x, pos))
            loss, loss_dict = loss_fn(y_pred, y)
            return x, y_pred, y, loss_dict

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer

def create_logdir(args):
    log_dir = Path(args.logdir) 
    hp_str = f"net={args.net}_targo_full"
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_dir / f"{hp_str}/{time_stamp}"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    args.logdir = log_dir
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TARGO Full with complete target point clouds")
    parser.add_argument("--net", default="targo", choices=["targo"], 
                        help="Network type: targo (original TARGO architecture)")
    parser.add_argument("--dataset", type=Path, default='/storage/user/dira/nips_data_version6/combined/targo_dataset')
    parser.add_argument("--data_contain", type=str, default="pc and targ_grid", help="Data content specification")
    parser.add_argument("--decouple", type=str2bool, default=False, help="Decouple flag")
    parser.add_argument("--dataset_raw", type=Path, default='/storage/user/dira/nips_data_version6/combined/targo_dataset')
    parser.add_argument("--logdir", type=Path, default="/usr/stud/dira/GraspInClutter/grasping/train_logs_targo_full")
    parser.add_argument("--description", type=str, default="targo_full_complete_target")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default='')
    parser.add_argument("--vis_data", type=str2bool, default=False, help="whether to visualize the dataset")
    parser.add_argument("--complete_shape", type=str2bool, default=True, help="use the complete the TSDF for grasp planning")
    parser.add_argument("--ablation_dataset", type=str, default='', help="1_10| 1_100| no_single_double | only_single_double|resized_set_theory|only_cluttered")
    parser.add_argument("--targ_grasp", type=str2bool, default=False, help="If true, use the target grasp mode, else use the clutter removal mode")
    parser.add_argument("--set_theory", type=str2bool, default=True, help="If true, use the target grasp mode, else use the clutter removal mode")
    parser.add_argument("--use_complete_targ", type=str2bool, default=True, help="Always True for TARGO Full")
    parser.add_argument("--lr-schedule-interval", type=int, default=10, help="Number of epochs between learning rate updates")
    parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate decay factor for scheduler")

    # Fixed parameters for TARGO Full
    parser.add_argument("--input_points", type=str, default='tsdf_points', help="Input point type")
    parser.add_argument("--shape_completion", type=str2bool, default=False, help="Always False for TARGO Full")

    args = parser.parse_args()
    
    # Force correct configuration for TARGO Full
    args.use_complete_targ = True
    args.shape_completion = False
    
    # Validate network type
    if args.net not in ["targo"]:
        raise ValueError("TARGO Full only supports original targo network type")
    
    # Print configuration
    print("=" * 60)
    print("TARGO Full Training Configuration:")
    print("=" * 60)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    print("\n" + "=" * 60)
    print("USING COMPLETE TARGET POINT CLOUDS (NO SHAPE COMPLETION)")
    print("ORIGINAL TARGO ARCHITECTURE")
    print("=" * 60)
    print("Make sure you have run the preprocessing script:")
    print("python scripts/preprocess_complete_target_mesh.py \\")
    print(f"  --raw_root {args.dataset_raw} \\")
    print(f"  --output_root {args.dataset}")
    print("=" * 60)

    create_logdir(args)
    print(f"\nLog directory: {args.logdir}")

    main(args) 