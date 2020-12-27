#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o out.%x
#SBATCH -e err.%x
#SBATCH -G 1
#SBATCH -J exp_convkernel_lossfunc
#SBATCH -a 0-100%1

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas
module load cuda10.1/blas/10.1.105
module load cudnn/10.1v7.6.5

# activate python environment
source ~/3dsegmentation/segmentation_env/bin/activate

log_path="$HOME/3dsegmentation/logs_exp_convkernel_lossfunc"

# choose hyperparameters
epochs=100
batch_size=2
weight_decay=0

# define architecture
depth=(6 6 6 6)
channels=(4 8 4 8)
conv_kernel=(5 5 3 3)
loss=('dice' 'dice' 'weighted_bce' 'weighted_bce')

d=$((depth[$SLURM_ARRAY_TASK_ID]))
ch=$((channels[$SLURM_ARRAY_TASK_ID]))
conv=$((conv_kernel[$SLURM_ARRAY_TASK_ID]))
loss=${loss[$SLURM_ARRAY_TASK_ID]}
st=1

# set folder name according to hyperparameters
experiment_title="unet_d${d}_ch${ch}_conv${conv}_loss_${loss}"

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
          --conv_kernel_size $conv --loss_type $loss \
          --MAX_PARAMS 50000000
fi

