#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

#python3 -u Code/train.py --save_name Isomag2D_ESRGAN --train_distributed True \
#--beta_1 0.9 --beta_2 0.999 \
#--num_workers 0 --data_folder Isomag2D --mode 2D \
#--cropping_resolution 256 --patch_size 1024 --training_patch_size 1024 \
#--num_blocks 3 --epochs 50 --random_flipping true \
#--min_dimension_size 32 --g_lr 0.0002 \
#--alpha_1 1.0 --alpha_2 0.00 \
#--model ESRGAN --generator_steps 1 

#python3 -u Code/train.py --save_name Isomag3D_ESRGAN --train_distributed True \
#--beta_1 0.9 --beta_2 0.999 \
#--num_workers 0 --data_folder Isomag3D --mode 3D \
#--cropping_resolution 96 --patch_size 96 --training_patch_size 96 \
#--num_blocks 3 --epochs 500 --random_flipping True \
#--min_dimension_size 32 --g_lr 0.00002 --alpha_1 1.0 --alpha_2 0.00 \
#--model ESRGAN --generator_steps 1

#python3 -u Code/train.py --save_name Mixing3D_ESRGAN --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Mixing3D --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --cropping_resolution 96 --min_dimension_size 32 \
#--g_lr 0.00002 --alpha_1 1.0 --alpha_2 0.00 \
#--model ESRGAN --generator_steps 1

#python3 -u Code/train.py --save_name Plume_ESRGAN --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Plume --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 750 --min_dimension_size 32 --cropping_resolution 96 --g_lr 0.00005 \
#--alpha_1 1.0 --alpha_2 0.00 \
#--model ESRGAN --generator_steps 1

#python3 -u Code/train.py --save_name Vorts_ESRGAN --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Vorts --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 500 --min_dimension_size 32 --cropping_resolution 96 --g_lr 0.00002 \
#--alpha_1 1.0 --alpha_2 0.00 \
#--model ESRGAN --generator_steps 1

python3 -u Code/train.py --save_name Supernova_ESRGAN --train_distributed True \
--beta_1 0.9 --beta_2 0.999 \
--num_workers 0 --data_folder Supernova --mode 3D \
--cropping_resolution 96 --patch_size 96 --training_patch_size 96 \
--num_blocks 3 --epochs 500 --random_flipping True \
--min_dimension_size 28 --g_lr 0.00002 --alpha_1 1.0 --alpha_2 0.00 \
--model ESRGAN --generator_steps 1