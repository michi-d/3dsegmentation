#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o out.%x
#SBATCH -e err.%x
#SBATCH -G gtx1080:1
#SBATCH -J diagonalsearch
#SBATCH -a 0-100%1

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas
module load cuda10.1/blas/10.1.105
module load cudnn/10.1v7.6.5

# activate python environment
source ~/3dsegmentation/segmentation_env/bin/activate

log_path="$HOME/3dsegmentation/logs_diagonalsearch"

# choose hyperparameters
epochs=100
batch_size=2
weight_decay=0

# define architecture
depth1M=(6 5 4 3)
channels1M=(4 8 16 32)
depth5M=(6 5 4)
channels5M=(8 16 32)
depth22M=(6 5)
channels22M=(16 32)

depth=("${depth1M[@]}" "${depth5M[@]}" "${depth22M[@]}")
channels=("${channels1M[@]}" "${channels5M[@]}" "${channels22M[@]}")

#depth1M=(2 1)
#channels1M=(64 128)
#depth5M=(3 2 1)
#channels5M=(64 128 256)
#depth22M=(4 3 2 1)
#channels22M=(64 128 256 512)
#depth=("${depth22M[@]}" "${depth5M[@]}" "${depth1M[@]}")
#channels=("${channels22M[@]}" "${channels5M[@]}" "${channels1M[@]}")

d=$((depth[$SLURM_ARRAY_TASK_ID]))
ch=$((channels[$SLURM_ARRAY_TASK_ID]))

num_stacks=1

# set folder name according to hyperparameters
experiment_title="unet_d${d}_ch${ch}"

if [ -d "${log_path}/${experiment_title}" ]; then
      echo "Experiment already done: ${experiment_title}"
      echo "Try new experiment..."
      :
else
   echo "Starting experiment: ${experiment_title}"
   python train_basic_3dunet.py --experiment_title $experiment_title \
          --log_path $log_path --batch_size $batch_size \
          --weight_decay $weight_decay \
          --arch stacked_unet --num_stacks $num_stacks \
          --epochs $epochs --depth $d --start_channels $ch \
          --MAX_PARAMS 50000000
fi

