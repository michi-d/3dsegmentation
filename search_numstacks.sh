#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o out.%x
#SBATCH -e err.%x
#SBATCH -G 1
#SBATCH -J search_numstacks
#SBATCH -a 0-100%1

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas
module load cuda10.1/blas/10.1.105
module load cudnn/10.1v7.6.5

# activate python environment
source ~/3dsegmentation/segmentation_env/bin/activate

log_path="$HOME/3dsegmentation/logs_search_numstacks"

# choose hyperparameters
epochs=100
batch_size=2
weight_decay=0

# define architecture
depth5M=(6 6 6)
channels5M=(8 8 8)
stacks5M=(1 2 3)

depth1M=(6 6 6 6 6)
channels1M=(4 4 4 4 4)
stacks1M=(1 2 4 6 8)

depth=("${depth5M[@]}" "${depth1M[@]}")
channels=("${channels5M[@]}" "${channels1M[@]}")
stacks=("${stacks5M[@]}" "${stacks1M[@]}")

d=$((depth[$SLURM_ARRAY_TASK_ID]))
ch=$((channels[$SLURM_ARRAY_TASK_ID]))
st=$((stacks[$SLURM_ARRAY_TASK_ID]))

# set folder name according to hyperparameters
experiment_title="unet_d${d}_ch${ch}_st${st}"

if [ -d "${log_path}/${experiment_title}" ]; then
      echo "Experiment already done: ${experiment_title}"
      echo "Try new experiment..."
      :
else
   echo "Starting experiment: ${experiment_title}"
   python train_basic_3dunet.py --experiment_title $experiment_title \
          --log_path $log_path --batch_size $batch_size \
          --weight_decay $weight_decay \
          --arch stacked_unet --num_stacks $st \
          --epochs $epochs --depth $d --start_channels $ch \
          --MAX_PARAMS 50000000
fi

