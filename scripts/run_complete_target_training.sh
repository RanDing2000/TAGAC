#!/bin/bash

# Complete Target Mesh Training Script for TARGO PTv3
# This script demonstrates how to use complete target meshes for training

set -e  # Exit on any error

# Configuration
RAW_DATASET="/storage/user/dira/nips_data_version6/combined/targo_dataset"
PROCESSED_DATASET="/storage/user/dira/nips_data_version6/combined/targo_dataset"
LOG_DIR="/usr/stud/dira/GraspInClutter/grasping/train_logs_targo_ptv3_complete"

echo "============================================================"
echo "Complete Target Mesh Training for TARGO PTv3"
echo "============================================================"

# Step 1: Check if preprocessing is needed
echo "Step 1: Checking if complete target mesh preprocessing is needed..."

# Check if any scene file contains complete target data
SAMPLE_SCENE=$(find ${PROCESSED_DATASET}/scenes -name "*.npz" | head -1)
if [ -f "$SAMPLE_SCENE" ]; then
    # Check if complete target data exists
    python3 -c "
import numpy as np
import sys
try:
    data = np.load('$SAMPLE_SCENE', allow_pickle=True)
    if 'complete_target_pc' in data:
        print('Complete target data found in dataset')
        sys.exit(0)
    else:
        print('Complete target data NOT found in dataset')
        sys.exit(1)
except Exception as e:
    print(f'Error checking dataset: {e}')
    sys.exit(1)
"
    PREPROCESSING_NEEDED=$?
else
    echo "No scene files found in dataset"
    PREPROCESSING_NEEDED=1
fi

# Step 2: Run preprocessing if needed
if [ $PREPROCESSING_NEEDED -eq 1 ]; then
    echo "Step 2: Running complete target mesh preprocessing..."
    echo "This may take a while depending on dataset size..."
    
    python scripts/preprocess_complete_target_mesh.py \
        --raw_root ${RAW_DATASET} \
        --output_root ${PROCESSED_DATASET} \
        --max_scenes 0
    
    echo "Preprocessing completed!"
else
    echo "Step 2: Preprocessing not needed - complete target data already exists"
fi

# Step 3: Load required modules
echo "Step 3: Loading required modules..."
module load compiler/gcc-8.3
module load cuda/11.3.0

# Step 4: Activate conda environment
echo "Step 4: Activating conda environment..."
conda activate targo

# Step 5: Start training with complete target meshes
echo "Step 5: Starting training with complete target meshes..."

python scripts/train_targo_ptv3.py \
    --net targo_ptv3 \
    --dataset ${PROCESSED_DATASET} \
    --dataset_raw ${RAW_DATASET} \
    --logdir ${LOG_DIR} \
    --data_contain "pc and targ_grid" \
    --use_complete_targ True \
    --shape_completion True \
    --epochs 50 \
    --batch-size 128 \
    --lr 2e-4 \
    --val-split 0.1 \
    --augment \
    --vis_data False \
    --description "complete_target_mesh_training"

echo "============================================================"
echo "Training completed!"
echo "Check logs at: ${LOG_DIR}"
echo "============================================================" 