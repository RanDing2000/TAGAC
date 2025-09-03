#!/bin/bash

# TARGO Full Training Script
# This script trains TARGO Full model using complete target point clouds without shape completion

set -e  # Exit on any error

# Configuration
RAW_DATASET="/home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset"
PROCESSED_DATASET="/home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset"
LOG_DIR="/home/ran.ding/projects/TARGO/train_logs_targo_full"

echo "============================================================"
echo "TARGO Full Training - Complete Target Point Clouds"
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

# Step 5: Start TARGO Full training
echo "Step 5: Starting TARGO Full training..."

python scripts/train_targo_full.py \
    --net targo_ptv3 \
    --dataset ${PROCESSED_DATASET} \
    --dataset_raw ${RAW_DATASET} \
    --logdir ${LOG_DIR} \
    --data_contain "pc and targ_grid" \
    --use_complete_targ True \
    --shape_completion False \
    --epochs 50 \
    --batch-size 32 \
    --lr 2e-4 \
    --val-split 0.1 \
    --augment \
    --vis_data False \
    --description "targo_full_complete_target_no_shape_completion" \
    --lr-schedule-interval 10 \
    --gamma 0.95

echo "============================================================"
echo "TARGO Full training completed!"
echo "Check logs at: ${LOG_DIR}"
echo "============================================================" 