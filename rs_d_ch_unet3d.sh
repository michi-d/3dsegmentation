#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o out.%x
#SBATCH -e err.%x
#SBATCH -G 1
#SBATCH -J rs_d_ch_unet3d

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas
module load cuda10.1/blas/10.1.105
module load cudnn/10.1v7.6.5

# activate python environment
source ~/3dsegmentation/segmentation_env/bin/activate

log_path="$HOME/3dsegmentation/logs_unet3d"

#iterations=$SLURM_ARRAY_TASK_ID
iterations=100

epochs=100
counter=1
while [ $counter -le $iterations ]
do
   # choose hyperparameters
   depth=$((1 + $RANDOM % 10))
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
             --log_path $log_path \
             --weight_decay $weight_decay \
             --epochs $epochs --depth $depth --start_channels $channels
             
      ((counter++))
   fi
done

