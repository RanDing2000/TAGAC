#!/bin/bash

# Example Usage Script for TARGO Training and Testing
# This script demonstrates how to use the new simplified training and testing interfaces

echo "============================================================"
echo "TARGO Training and Testing Example Usage"
echo "============================================================"

# Environment Setup
echo "Setting up environment..."
# conda activate targo
# module load compiler/gcc-8.3
# module load cuda/11.3.0

echo ""
echo "============================================================"
echo "1. TRAINING EXAMPLES"
echo "============================================================"

echo ""
echo "1.1 Quick training with 1/10 data for testing:"
echo "python scripts/train_simple.py --model_type targo --epochs 10 --ablation_dataset 1_10"

echo ""
echo "1.2 Train TARGO with shape completion:"
echo "python scripts/train_simple.py --model_type targo --epochs 50 --batch-size 32 --augment"

echo ""
echo "1.3 Train TARGO Full with complete target point clouds:"
echo "python scripts/train_simple.py --model_type targo_full --epochs 50 --batch-size 32 --augment"

echo ""
echo "1.4 Train with custom parameters:"
echo "python scripts/train_simple.py \\"
echo "    --model_type targo_full \\"
echo "    --epochs 100 \\"
echo "    --batch-size 64 \\"
echo "    --lr 1e-4 \\"
echo "    --augment \\"
echo "    --dataset /path/to/dataset \\"
echo "    --logdir /path/to/logs"

echo ""
echo "============================================================"
echo "2. TESTING EXAMPLES"
echo "============================================================"

echo ""
echo "2.1 Quick test with 1/10 scenes on YCB dataset:"
echo "python scripts/test_simple.py \\"
echo "    --model_type targo \\"
echo "    --model_path checkpoints/targonet.pt \\"
echo "    --dataset_type ycb \\"
echo "    --occlusion_level medium \\"
echo "    --subset_ratio 0.1"

echo ""
echo "2.2 Test TARGO Full on YCB dataset:"
echo "python scripts/test_simple.py \\"
echo "    --model_type targo_full \\"
echo "    --model_path checkpoints/targo_full.pt \\"
echo "    --dataset_type ycb \\"
echo "    --occlusion_level medium \\"
echo "    --subset_ratio 0.1"

echo ""
echo "2.3 Test on ACRONYM dataset:"
echo "python scripts/test_simple.py \\"
echo "    --model_type targo \\"
echo "    --model_path checkpoints/targonet.pt \\"
echo "    --dataset_type acronym \\"
echo "    --occlusion_level medium \\"
echo "    --subset_ratio 0.1"

echo ""
echo "2.4 Test with different subset ratios (5% of scenes):"
echo "python scripts/test_simple.py \\"
echo "    --model_type targo \\"
echo "    --model_path checkpoints/targonet.pt \\"
echo "    --dataset_type ycb \\"
echo "    --occlusion_level medium \\"
echo "    --subset_ratio 0.05"

echo ""
echo "2.5 Test with visualization enabled:"
echo "python scripts/test_simple.py \\"
echo "    --model_type targo \\"
echo "    --model_path checkpoints/targonet.pt \\"
echo "    --dataset_type ycb \\"
echo "    --occlusion_level medium \\"
echo "    --subset_ratio 0.1 \\"
echo "    --vis \\"
echo "    --video_recording True"

echo ""
echo "============================================================"
echo "3. PREPROCESSING EXAMPLES"
echo "============================================================"

echo ""
echo "3.1 Preprocess complete target meshes for TARGO Full:"
echo "python scripts/preprocess_complete_target_mesh.py \\"
echo "    --raw_root /home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset \\"
echo "    --output_root /home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset"

echo ""
echo "3.2 Verify preprocessing results:"
echo "python scripts/verify_complete_target_preprocessing.py \\"
echo "    --dataset_root /home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset \\"
echo "    --raw_root /home/ran.ding/projects/TARGO/data/nips_data_version6/combined/targo_dataset"

echo ""
echo "============================================================"
echo "4. AUTOMATED WORKFLOWS"
echo "============================================================"

echo ""
echo "4.1 Automated TARGO Full training (with preprocessing):"
echo "chmod +x scripts/run_targo_full_training.sh"
echo "./scripts/run_targo_full_training.sh"

echo ""
echo "============================================================"
echo "5. FULL EVALUATION EXAMPLES"
echo "============================================================"

echo ""
echo "5.1 Full YCB evaluation (all scenes):"
echo "python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium"
echo "python scripts/inference_ycb.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level medium"

echo ""
echo "5.2 Full ACRONYM evaluation (all scenes):"
echo "python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium"
echo "python scripts/inference_acronym.py --type targo_full_targ --model 'checkpoints/targo_full.pt' --occlusion-level medium"

echo ""
echo "============================================================"
echo "6. PARALLEL EVALUATION"
echo "============================================================"

echo ""
echo "6.1 Run parallel evaluation:"
echo "chmod +x ./scripts/eval_ycb_parallel.sh"
echo "./scripts/eval_ycb_parallel.sh"

echo ""
echo "6.2 Monitor parallel jobs:"
echo "tmux list-windows -t eval_parallel_ycb"
echo "tmux attach-session -t eval_parallel_ycb \\; select-window -t job_3"

echo ""
echo "============================================================"
echo "Example Usage Script Complete!"
echo "============================================================"
echo ""
echo "To run any of these commands, copy and paste them into your terminal."
echo "Make sure to:"
echo "1. Activate the conda environment: conda activate targo"
echo "2. Load required modules: module load compiler/gcc-8.3 && module load cuda/11.3.0"
echo "3. Adjust paths according to your setup"
echo ""
 