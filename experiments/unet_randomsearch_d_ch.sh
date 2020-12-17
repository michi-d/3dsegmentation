#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o out.%x
#SBATCH -e err.%x
#SBATCH -G 1
#SBATCH -J unet_randomsearch_d_ch
#SBATCH -C local

module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas
module load cuda10.1/blas/10.1.105
module load cudnn/10.1v7.6.5

# activate python environment
source ~/3dsegmentation/segmentation_env/bin/activate

log_path="$TMP_LOCAL/logs"
iterations=1

epochs=100
L_train=4096
L_validate=256
L_test=256

counter=1
while [ $counter -le $iterations ]
do
   depth=$((1 + $RANDOM % 10))
   channels=$((2**($RANDOM % 9)))
   depth=1
   channels=4
   epochs=2

   experiment_title="unet_d${depth}_ch${channels}"
   echo "$experiment_title"

   python experiments/train_basic_unet.py --experiment_title $experiment_title \
            --log_path $log_path \
            --L_train $L_train --L_validate $L_validate --L_test $L_test \
            --epochs $epochs --depth $depth --start_channels $channels

   ((counter++))
done
