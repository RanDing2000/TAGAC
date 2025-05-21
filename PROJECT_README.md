# Project Data and Model Guide

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

```
## Environment Setup
conda activate targo
module load cuda/11.3.0

## TARGONet, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/targo
python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium

## TARGONet_hunyun2, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/targo_hunyun2
python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level medium

## GIGA, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/giga
python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level no
python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level slight
python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level medium

## VGN, results saved to e.g. targo_eval_results/ycb/eval_results_full-medium-occlusion/vgn
python scripts/inference_ycb.py --type giga --model 'checkpoints/vgn_packed.pt' --occlusion-level no
python scripts/inference_ycb.py --type giga --model 'checkpoints/vgn_packed.pt' --occlusion-level slight
python scripts/inference_ycb.py --type giga --model 'checkpoints/vgn_packed.pt' --occlusion-level medium
```

- **Acronym Testing Script**: `scripts/inference_acronym.py`

```
## Environment Setup
conda activate targo
module load cuda/11.3.0

## TARGONet, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/targo
python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium

## TARGONet_hunyun2, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/targo_hunyun2
python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level no
python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level slight
python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level medium

## GIGA, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/giga
python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level no
python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level slight
python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level medium

## VGN, results saved to e.g. targo_eval_results/acronym/eval_results_full-medium-occlusion/vgn
python scripts/inference_acronym.py --type giga --model 'checkpoints/vgn_packed.pt' --occlusion-level no
python scripts/inference_acronym.py --type giga --model 'checkpoints/vgn_packed.pt' --occlusion-level slight
python scripts/inference_acronym.py --type giga --model 'checkpoints/vgn_packed.pt' --occlusion-level medium
```

### 2. Hunyuan3D Model [/usr/stud/dira/GraspInClutter/Gen3DSR]

First setup the environment refers to INSTALL.md

- **YCB Script**: `src/work/shape_completion_targo_ycb_amodal_icp_only_gt.py`
- **ACRONYM Script**: `src/work/shape_completion_targo_acronym_amodal_icp_only_gt.py`

# 2025.4.30

- 1. Some results on acronym medium dataset.
- 2. Some results on ycb medium dataset.
- 3. Some progress on end-to-end pipeline.
- 4. Some progress on FGC-Graspnet and anygrasp
- 5. Discussion on story