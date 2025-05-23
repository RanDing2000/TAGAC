#!/bin/bash

SESSION="eval_parallel_ycb"
WORKDIR="/usr/stud/dira/GraspInClutter/targo"
ENV_ACTIVATE="conda activate targo"
CUDA_MODULE="module load cuda/11.3.0"

# 启动新的 tmux 会话（后台运行）
tmux new-session -d -s $SESSION

# 所有命令列表
commands=(
  "python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level no"
  "python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level slight"
  "python scripts/inference_ycb.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium"

  "python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level no"
  "python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level slight"
  "python scripts/inference_ycb.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level medium"

  "python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level no"
  "python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level slight"
  "python scripts/inference_ycb.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level medium"

  "python scripts/inference_ycb.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level no"
  "python scripts/inference_ycb.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level slight"
  "python scripts/inference_ycb.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level medium"
)

# 创建窗口并执行每个命令
for cmd in "${commands[@]}"; do
  # 提取 type 和 occlusion-level 作为窗口名
  type=$(echo "$cmd" | grep -oP '(?<=--type )\S+')
  occ=$(echo "$cmd" | grep -oP '(?<=--occlusion-level )\S+')
  window_name="${type}_${occ}"

  tmux new-window -t $SESSION -n $window_name
  tmux send-keys -t $SESSION:$window_name "$CUDA_MODULE; source ~/.bashrc; $ENV_ACTIVATE; cd $WORKDIR; $cmd; echo '✅ Done: $cmd'; sleep 2; exit" C-m
done

echo "Launched ${#commands[@]} parallel jobs in tmux session '$SESSION'."
echo "Use 'tmux attach -t $SESSION' to monitor progress."
