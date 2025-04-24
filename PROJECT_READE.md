# Project Data and Model Guide

## Data Processing Workflow

### 1. Scene Data Generation [/usr/stud/dira/GraspInClutter/targo]

#### Acronym Dataset
- **Medium Occlusion Scenes**: `scripts/scene_generation/acronym_objects/generate_targo_dataset_acroym_middle_occlusion.py`
- **Slight Occlusion Scenes**: `scripts/scene_generation/acronym_objects/generate_targo_dataset_acroym_slight_occlusion.py`
- **No Occlusion Scenes**: `scripts/scene_generation/acronym_objects/generate_targo_dataset_acroym_no_occlusion.py`

#### YCB Dataset
- **Medium Occlusion Scenes**: `scripts/scene_generation/ycb_objects/generate_targo_dataset_ycb_middle_occlusion.py`
- **Slight Occlusion Scenes**: `scripts/scene_generation/ycb_objects/generate_targo_dataset_ycb_slight_occlusion.py`
- **No Occlusion Scenes**: `scripts/scene_generation/ycb_objects/generate_targo_dataset_ycb_no_occlusion.py`

### 2. Scene Data Rendering [/usr/stud/dira/GraspInClutter/GaussianGrasp/datasets_gen]
- **Acronym Rendering Script**: `scripts/work/make_rgb_dataset_acronym.py`
- **YCB Rendering Script**: `scripts/work/make_rgb_dataset_ycb.py`

> **Note**: Path configuration needs to be updated in `/usr/stud/dira/GraspInClutter/GaussianGrasp/centergrasp/configs.py`

## Model Evaluation and Demonstration

### 1. Benchmarking [/usr/stud/dira/GraspInClutter/targo]
- **YCB Testing Script**: `scripts/inference_ycb.py`
- **Acronym Testing Script**: `scripts/inference_acronym.py`

### 2. Hunyuan3D Model [/usr/stud/dira/GraspInClutter/Gen3DSR]
First setup the environment refers to INSTALL.md
- **YCB Script**: `src/work/shape_completion_targo_ycb_amodal_icp_only_gt.py`
- **ACRONYM Script**: `src/work/shape_completion_targo_acronym_amodal_icp_only_gt.py`