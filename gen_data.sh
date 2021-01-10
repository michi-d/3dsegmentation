#!/bin/bash

#SBATCH -p fat
#SBATCH -t 2-00:00:00
#SBATCH -o out.%x
#SBATCH -e err.%x
#SBATCH -J gen_data

# activate python environment
source ~/3dsegmentation/segmentation_env/bin/activate

python gen_data.py --filename vol_train_set1.h5 --L 512 --random_state 0 \
                   --NF_size 10 --std_intensity_range_low 0.3 --std_intensity_range_high 1.2

python gen_data.py --filename vol_train_set2.h5 --L 512 --random_state 512 \
                   --NF_size 10 --std_intensity_range_low 0.3 --std_intensity_range_high 1.2

python gen_data.py --filename vol_train_set3.h5 --L 512 --random_state 1024 \
                   --NF_size 10 --std_intensity_range_low 0.3 --std_intensity_range_high 1.2

python gen_data.py --filename vol_train_set4.h5 --L 512 --random_state 1536 \
                   --NF_size 10 --std_intensity_range_low 0.3 --std_intensity_range_high 1.2

python gen_data.py --filename vol_val_set.h5 --L 64 --random_state 9999 \
                   --NF_size 10 --std_intensity_range_low 0.3 --std_intensity_range_high 1.2
