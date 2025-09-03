#!/bin/bash
#SBATCH --job-name=eval_acronym
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:p6000:1
#SBATCH --time=3-12:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/%x-%j.out

# 安全设置：出错时退出
set -e

# 启动环境
module load cuda/11.3.0
source ~/.bashrc
conda activate targo

# 设置变量
SESSION="eval_parallel_acronym2"
WORKDIR="/home/ran.ding/projects/TARGO"

# 创建 tmux 会话（后台）
tmux new-session -d -s $SESSION

# 所有命令列表
commands=(
  "python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level no"
  "python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level slight"
  "python scripts/inference_acronym.py --type targo --model 'checkpoints/targonet.pt' --occlusion-level medium"

  "python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level no"
  "python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level slight"
  "python scripts/inference_acronym.py --type targo_hunyun2 --model 'checkpoints/targonet.pt' --occlusion-level medium"

  "python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level no"
  "python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level slight"
  "python scripts/inference_acronym.py --type giga --model 'checkpoints/giga_packed.pt' --occlusion-level medium"

  "python scripts/inference_acronym.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level no"
  "python scripts/inference_acronym.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level slight"
  "python scripts/inference_acronym.py --type vgn --model 'checkpoints/vgn_packed.pt' --occlusion-level medium"
)

# 为每条命令创建窗口并执行
for cmd in "${commands[@]}"; do
  type=$(echo "$cmd" | grep -oP '(?<=--type )\S+')
  occ=$(echo "$cmd" | grep -oP '(?<=--occlusion-level )\S+')
  window_name="${type}_${occ}"

  tmux new-window -t $SESSION -n $window_name
  tmux send-keys -t $SESSION:$window_name "cd $WORKDIR; $cmd; echo '✅ Done: $cmd'; sleep 2; exit" C-m
done

echo "Launched ${#commands[@]} jobs in tmux session '$SESSION'"
