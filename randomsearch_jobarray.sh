#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o out.%x
#SBATCH -e err.%x
#SBATCH -G 1
#SBATCH -J randomsearch
#SBATCH -a 0-100%1

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas
module load cuda10.1/blas/10.1.105
module load cudnn/10.1v7.6.5

# activate python environment
source ~/3dsegmentation/segmentation_env/bin/activate

log_path="$HOME/3dsegmentation/logs_unet3d_bs2"

# choose hyperparameters
epochs=100

depth=$((1 + $RANDOM % 7))
channels=$((2**($RANDOM % 9)))

if [ $(($RANDOM % 2)) -eq 0 ]; then
  weight_decay=0
else
  weight_decay=1e-2
fi

# set folder name according to hyperparameters
experiment_title="unet_d${depth}_ch${channels}_wd${weight_decay}"

if [ -d "${log_path}/${experiment_title}" ]; then
      echo "Experiment already done: ${experiment_title}"
      echo "Try new combination..."
      :
else
   echo "Starting experiment: ${experiment_title}"
   python train_basic_3dunet.py --experiment_title $experiment_title \
          --log_path $log_path --batch_size 2 \
          --weight_decay $weight_decay \
          --epochs $epochs --depth $depth --start_channels $channels
fi

