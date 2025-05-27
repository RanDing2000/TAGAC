# Project Data and Model Guide

## Model Architecture Overview

### TARGO Model Variants

#### Original TARGO Models
- **TARGO**: Original TARGO model with shape completion using AdaPoinTr
- **TARGO Full**: Original TARGO model using preprocessed complete target point clouds (no shape completion)

#### PointTransformerV3-based TARGO Variants  
- **TARGO PTV3** (`targo_ptv3`): TARGO variant based on PointTransformerV3 architecture with shape completion
- **PTV3 Scene** (`ptv3_scene`): PointTransformerV3-based variant for scene understanding

## Designing new network
### 1. generate complete target mesh for training the network
```scripts/preprocess_complete_target_mesh.py
```

## Model Training

### 1. Simple Training Interface

We provide a unified training interface for both original TARGO and PointTransformerV3-based variants:

#### For Original TARGO Models (targo, targo_full)
```bash
# Environment Setup for Original TARGO
conda activate targo
module load compiler/gcc-8.3
module load cuda/11.3.0

# Train original TARGO with shape completion
python scripts/train_simple.py --model_type targo --epochs 50 --batch-size 32 --augment

# Train original TARGO Full with complete target point clouds (no shape completion)
python scripts/train_simple.py --model_type targo_full --epochs 50 --batch-size 32 --augment

# Train with dataset ablation (1/10 of data)
python scripts/train_simple.py --model_type targo --epochs 50 --ablation_dataset 1_10

# Train with custom parameters
python scripts/train_simple.py \
    --model_type targo_full \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --augment \
    --dataset /path/to/dataset \
    --logdir /path/to/logs
```

#### For PointTransformerV3-based Variants (targo_ptv3, ptv3_scene)
```bash
# Environment Setup for PointTransformerV3 Variants
conda activate ptv3
module load cuda/12.1.0

# Train PointTransformerV3-based TARGO variant
python scripts/train_targo_ptv3.py --net targo_ptv3 --epochs 50 --batch-size 32 --augment

# Train PTV3 Scene variant
python scripts/train_ptv3_scene.py --net ptv3_scene --epochs 50 --batch-size 32 --augment
```

### 2. Individual Training Scripts

#### Original TARGO with Shape Completion
```bash
# Environment Setup
conda activate targo
module load compiler/gcc-8.3
module load cuda/11.3.0

python scripts/train_targo.py \
    --net targo \
    --dataset /storage/user/dira/nips_data_version6/combined/targo_dataset \
    --dataset_raw /storage/user/dira/nips_data_version6/combined/targo_dataset \
    --logdir /usr/stud/dira/GraspInClutter/grasping/train_logs_targo \
    --data_contain "pc and targ_grid" \
    --use_complete_targ False \
    --shape_completion True \
    --epochs 50 \
    --batch-size 32 \
    --lr 2e-4 \
    --augment
```

#### PointTransformerV3-based TARGO Variant
```bash
# Environment Setup
conda activate ptv3
module load cuda/12.1.0

python scripts/train_targo_ptv3.py \
    --net targo_ptv3 \
    --dataset /storage/user/dira/nips_data_version6/combined/targo_dataset \
    --dataset_raw /storage/user/dira/nips_data_version6/combined/targo_dataset \
    --logdir /usr/stud/dira/GraspInClutter/grasping/train_logs_targo_ptv3 \
    --data_contain "pc and targ_grid" \
    --use_complete_targ False \
    --shape_completion True \
    --epochs 50 \
    --batch-size 32 \
    --lr 2e-4 \
    --augment
```

#### Original TARGO Full with Complete Target Point Clouds
```bash
# Environment Setup
conda activate targo
module load compiler/gcc-8.3
module load cuda/11.3.0

# First, preprocess complete target meshes
python scripts/preprocess_complete_target_mesh.py \
    --raw_root /storage/user/dira/nips_data_version6/combined/targo_dataset \
    --output_root /storage/user/dira/nips_data_version6/combined/targo_dataset

# Then train original TARGO Full
python scripts/train_targo_full.py \
    --net targo \
    --dataset /storage/user/dira/nips_data_version6/combined/targo_dataset \
    --dataset_raw /storage/user/dira/nips_data_version6/combined/targo_dataset \
    --logdir /usr/stud/dira/GraspInClutter/grasping/train_logs_targo_full \
    --data_contain "pc and targ_grid" \
    --use_complete_targ True \
    --shape_completion False \
    --epochs 50 \
    --batch-size 32 \
    --lr 2e-4 \
    --augment
```

#### Automated Training with Preprocessing
```bash
# Use the automated script that handles preprocessing
chmod +x scripts/run_targo_full_training.sh
./scripts/run_targo_full_training.sh
```

## Model Testing

### 1. Simple Testing Interface (1/10 Scene Sampling)

For quick evaluation, we provide a simple test script that samples 1/10 of the test scenes:

#### For Original TARGO Models
```bash
# Environment Setup for Original TARGO
conda activate targo
module load cuda/11.3.0

# Test original TARGO on YCB dataset (medium occlusion, 1/10 scenes)
python scripts/test_simple.py \
    --model_type targo \
    --model_path checkpoints/targonet.pt \
    --dataset_type ycb \
    --occlusion_level medium \
    --subset_ratio 0.1

# Test original TARGO Full on YCB dataset
python scripts/test_simple.py \
    --model_type targo_full \
    --model_path checkpoints/targo_full.pt \
    --dataset_type ycb \
    --occlusion_level medium \
    --subset_ratio 0.1

# Test on ACRONYM dataset
python scripts/test_simple.py \
    --model_type targo \
    --model_path checkpoints/targonet.pt \
    --dataset_type acronym \
    --occlusion_level medium \
    --subset_ratio 0.1

# Test with different subset ratios
python scripts/test_simple.py \
    --model_type targo \
    --model_path checkpoints/targonet.pt \
    --dataset_type ycb \
    --occlusion_level medium \
    --subset_ratio 0.05  # 5% of scenes

# Test with visualization enabled
python scripts/test_simple.py \
    --model_type targo \
    --model_path checkpoints/targonet.pt \
    --dataset_type ycb \
    --occlusion_level medium \
    --subset_ratio 0.1 \
    --vis \
    --video_recording True
```

#### For PointTransformerV3-based Variants
```bash
# Environment Setup for PointTransformerV3 Variants
conda activate ptv3
module load cuda/12.1.0

# Test PointTransformerV3-based TARGO variant on YCB dataset
python scripts/inference_ycb.py \
    --type targo_ptv3 \
    --model checkpoints/targo_ptv3.pt \
    --occlusion-level medium

# Test PTV3 Scene variant
python scripts/inference_ycb.py \
    --type ptv3_scene \
    --model checkpoints/ptv3_scene.pt \
    --occlusion-level medium
```

### 2. Full Testing (All Scenes)

For complete evaluation, use the original inference scripts:

#### YCB Dataset Testing
```bash
## Environment Setup for Original TARGO
conda activate targo
module load cuda/11.3.0

## Original TARGONet, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/targo
python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium

## Original TARGONet_full_targ (TARGO Full), results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/targo_full_targ
python scripts/inference_ycb.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level no
python scripts/inference_ycb.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level slight
python scripts/inference_ycb.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level medium
```

```bash
## Environment Setup for PointTransformerV3 Variants
conda activate ptv3
module load cuda/12.1.0

## PointTransformerV3-based TARGONet, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/targo_ptv3
python scripts/inference_ycb.py --type targo_ptv3 --model 'checkpoints/targo_ptv3.pt' --occlusion-level no
python scripts/inference_ycb.py --type targo_ptv3 --model 'checkpoints/targo_ptv3.pt' --occlusion-level slight
python scripts/inference_ycb.py --type targo_ptv3 --model 'checkpoints/targo_ptv3.pt' --occlusion-level medium

## PTV3 Scene variant
python scripts/inference_ycb.py --type ptv3_scene --model 'checkpoints/ptv3_scene.pt' --occlusion-level no
python scripts/inference_ycb.py --type ptv3_scene --model 'checkpoints/ptv3_scene.pt' --occlusion-level slight
python scripts/inference_ycb.py --type ptv3_scene --model 'checkpoints/ptv3_scene.pt' --occlusion-level medium
```

```bash
## Other Models (use original TARGO environment)
conda activate targo
module load cuda/11.3.0

## TARGONet_hunyun2, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/targo_hunyun2
python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level medium

## GIGA, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/giga
python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level no
python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level slight
python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level medium

## VGN, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/vgn
python scripts/inference_ycb.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level no
python scripts/inference_ycb.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level slight
python scripts/inference_ycb.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level medium
```

#### ACRONYM Dataset Testing
```bash
## Environment Setup for Original TARGO
conda activate targo
module load cuda/11.3.0

## Original TARGONet, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/targo
python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium

## Original TARGONet_full_targ (TARGO Full), results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/targo_full_targ
python scripts/inference_acronym.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level no
python scripts/inference_acronym.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level slight
python scripts/inference_acronym.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level medium
```

```bash
## Environment Setup for PointTransformerV3 Variants
conda activate ptv3
module load cuda/12.1.0

## PointTransformerV3-based TARGONet, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/targo_ptv3
python scripts/inference_acronym.py --type targo_ptv3 --model 'checkpoints/targo_ptv3.pt' --occlusion-level no
python scripts/inference_acronym.py --type targo_ptv3 --model 'checkpoints/targo_ptv3.pt' --occlusion-level slight
python scripts/inference_acronym.py --type targo_ptv3 --model 'checkpoints/targo_ptv3.pt' --occlusion-level medium

## PTV3 Scene variant
python scripts/inference_acronym.py --type ptv3_scene --model 'checkpoints/ptv3_scene.pt' --occlusion-level no
python scripts/inference_acronym.py --type ptv3_scene --model 'checkpoints/ptv3_scene.pt' --occlusion-level slight
python scripts/inference_acronym.py --type ptv3_scene --model 'checkpoints/ptv3_scene.pt' --occlusion-level medium
```

```bash
## Other Models (use original TARGO environment)
conda activate targo
module load cuda/11.3.0

## TARGONet_hunyun2, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/targo_hunyun2
python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level medium

## GIGA, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/giga
python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level no
python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level slight
python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level medium

## VGN, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/vgn
python scripts/inference_acronym.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level no
python scripts/inference_acronym.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level slight
python scripts/inference_acronym.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level medium
```

## Data Processing Workflow

### 1. Scene Data Generation (targo) [/usr/stud/dira/GraspInClutter/targo]

#### Acronym Dataset

- **Medium Occlusion Scenes**: `scripts/scene_generation/acronym_objects/generate_targo_dataset_acroym_medium_occlusion.py`

  The data is saved on `/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym-middle-occlusion-1000`

  Also, since the mesh_pose_dict has been corrupted, we use:

  `scripts/scene_generation/acronym_objects/generate_targo_dataset_acroym_middle_occlusion_from_known.py`

- **Slight Occlusion Scenes**: `scripts/scene_generation/acronym_objects/generate_targo_dataset_acroym_slight_occlusion.py`

  The data is saved on `/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym-slight-occlusion-1000`

- **No Occlusion Scenes**: `scripts/scene_generation/acronym_objects/generate_targo_dataset_acroym_no_occlusion.py`

  The data is saved on `/usr/stud/dira/GraspInClutter/targo/data_scenes/acronym/acronym-no-occlusion-1000`

#### YCB Dataset

- **Medium Occlusion Scenes**: `scripts/scene_generation/ycb_objects/generate_targo_dataset_ycb_medium_occlusion.py`

  The data is saved on `/usr/stud/dira/GraspInClutter/targo/data_scenes/ycb/maniskill-ycb-v2-middle-occlusion-1000`

- **Slight Occlusion Scenes**: `scripts/scene_generation/ycb_objects/generate_targo_dataset_ycb_slight_occlusion.py`

  The data is saved on `/usr/stud/dira/GraspInClutter/targo/data_scenes/ycb/maniskill-ycb-v2-slight-occlusion-1000`

- **No Occlusion Scenes**: `scripts/scene_generation/ycb_objects/generate_targo_dataset_ycb_no_occlusion.py`

  The data is saved on `/usr/stud/dira/GraspInClutter/targo/data_scenes/ycb/maniskill-ycb-v2-no-occlusion-1000`

### 2. Scene Data Rendering (GaussianGrasp) [/usr/stud/dira/GraspInClutter/GaussianGrasp/datasets_gen]

- **Acronym Rendering Script**: `scripts/work/make_rgb_dataset_acronym.py`
- **YCB Rendering Script**: `scripts/work/make_rgb_dataset_ycb.py`

### 3. Dataset Stastics (Gen3DSR) [/usr/stud/dira/GraspInClutter/Gen3DSR]

- **Acronym Prompt Dict**: `data/acronym/acronym_prompt_dict.json`
- **YCB Prompt Dict**: `data/ycb/ycb_prompt_dict.json`

> **Note**: Path configuration needs to be updated in `/usr/stud/dira/GraspInClutter/GaussianGrasp/centergrasp/configs.py`

## Model Evaluation and Demonstration

### 1. Benchmarking [/usr/stud/dira/GraspInClutter/targo]

- **YCB Testing Script**: `scripts/inference_ycb.py`
- **ACRONYM Testing Script**: `scripts/inference_acronym.py`
- **Simple Testing Script (1/10 scenes)**: `scripts/test_simple.py`

### 2. Hunyuan3D Model [/usr/stud/dira/GraspInClutter/Gen3DSR]

First setup the environment refers to INSTALL.md

- **YCB Script**: `src/work/shape_completion_targo_ycb_amodal_icp_only_gt.py`
- **ACRONYM Script**: `src/work/shape_completion_targo_acronym_amodal_icp_only_gt.py`

## Quick Start Examples

### Training Examples
```bash
# Quick training with 1/10 data for testing (original TARGO)
conda activate targo && module load compiler/gcc-8.3 && module load cuda/11.3.0
python scripts/train_simple.py --model_type targo --epochs 10 --ablation_dataset 1_10

# Full training for original TARGO
conda activate targo && module load compiler/gcc-8.3 && module load cuda/11.3.0
python scripts/train_simple.py --model_type targo --epochs 50 --augment

# Full training for original TARGO Full
conda activate targo && module load compiler/gcc-8.3 && module load cuda/11.3.0
python scripts/train_simple.py --model_type targo_full --epochs 50 --augment

# Training PointTransformerV3-based TARGO variant
conda activate ptv3 && module load cuda/12.1.0
python scripts/train_targo_ptv3.py --net targo_ptv3 --epochs 50 --augment
```

### Testing Examples
```bash
# Quick test with 1/10 scenes (original TARGO)
conda activate targo && module load cuda/11.3.0
python scripts/test_simple.py --model_type targo --model_path checkpoints/targonet.pt --dataset_type ycb --occlusion_level medium

# Quick test with visualization (original TARGO)
conda activate targo && module load cuda/11.3.0
python scripts/test_simple.py --model_type targo --model_path checkpoints/targonet.pt --dataset_type ycb --occlusion_level medium --vis

# Test different subset ratios (original TARGO)
conda activate targo && module load cuda/11.3.0
python scripts/test_simple.py --model_type targo --model_path checkpoints/targonet.pt --dataset_type ycb --occlusion_level medium --subset_ratio 0.05

# Test PointTransformerV3-based TARGO variant
conda activate ptv3 && module load cuda/12.1.0
python scripts/inference_ycb.py --type targo_ptv3 --model checkpoints/targo_ptv3.pt --occlusion-level medium
```

# 2025.4.30

- 1. Some results on acronym medium dataset.
- 2. Some results on ycb medium dataset.
- 3. Some progress on end-to-end pipeline.
- 4. Some progress on FGC-Graspnet and anygrasp
- 5. Discussion on story