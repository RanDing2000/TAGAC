#!/bin/bash

# TARGO-PTv3 Training Script
# This script trains the new TARGO model with PointTransformerV3 encoder

# Environment setup
source ~/.bashrc
conda activate targo
module load compiler/gcc-8.3
module load cuda/11.3.0

# Navigate to project directory
cd /home/ran.ding/projects/TARGO

# Training parameters
DATASET_PATH="/home/ran.ding/projects/TARGO/data/nips_data_version6/combined"
LOG_DIR="/home/ran.ding/projects/TARGO/train_logs_targo_ptv3"
BATCH_SIZE=2
EPOCHS=5
LEARNING_RATE=2e-4

echo "Starting TARGO-PTv3 training..."
echo "Dataset: $DATASET_PATH"
echo "Log directory: $LOG_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"

# Run training
python scripts/train_targo_ptv3.py \
    --net targo_ptv3 \
    --dataset $DATASET_PATH \
    --dataset_raw $DATASET_PATH \
    --logdir $LOG_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --val-split 0.1 \
    --augment \
    --shape_completion True \
    --input_points tsdf_points \
    --sc_model_config "/home/ran.ding/projects/TARGO/src/shape_completion/configs/stso/AdaPoinTr.yaml" \
    --sc_model_checkpoint "/home/ran.ding/projects/TARGO/checkpoints_gaussian/sc_net/ckpt-best_0425.pth" \
    --description "targo_ptv3_test_run"

echo "Training completed!" 