#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 4 --nodes_per_layer 256 \
#--save_name isotropic_coarse_vf_512_4x256 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:1 --data_device cuda:0 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 4 --nodes_per_layer 512 \
#--save_name isotropic_coarse_vf_512_4x512 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:2 --data_device cuda:1 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 4 --nodes_per_layer 1024 \
#--save_name isotropic_coarse_vf_512_4x1024 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:3 --data_device cuda:2 &

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 4 --nodes_per_layer 2048 \
--save_name isotropic_coarse_vf_512_4x2048 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:0 &

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 4 --nodes_per_layer 4096 \
--save_name isotropic_coarse_vf_512_4x4096 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:4 &


###########################################################

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 6 --nodes_per_layer 256 \
#--save_name isotropic_coarse_vf_512_6x256 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:5 --data_device cuda:4 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 6 --nodes_per_layer 512 \
#--save_name isotropic_coarse_vf_512_6x512 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:6 --data_device cuda:5 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 6 --nodes_per_layer 1024 \
#--save_name isotropic_coarse_vf_512_6x1024 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:7 --data_device cuda:6

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 6 --nodes_per_layer 2048 \
--save_name isotropic_coarse_vf_512_6x2048 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:7 --data_device cuda:1 &

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 6 --nodes_per_layer 4096 \
--save_name isotropic_coarse_vf_512_6x4096 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:7 --data_device cuda:5 &

###########################################################

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 8 --nodes_per_layer 256 \
#--save_name isotropic_coarse_vf_512_8x256 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:1 --data_device cuda:0 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 8 --nodes_per_layer 512 \
#--save_name isotropic_coarse_vf_512_8x512 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:2 --data_device cuda:1 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 8 --nodes_per_layer 1024 \
#--save_name isotropic_coarse_vf_512_8x1024 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:3 --data_device cuda:2 &

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 8 --nodes_per_layer 2048 \
--save_name isotropic_coarse_vf_512_8x2048 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:2 &

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 8 --nodes_per_layer 4096 \
--save_name isotropic_coarse_vf_512_8x4096 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:6 &

###########################################################


#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 10 --nodes_per_layer 256 \
#--save_name isotropic_coarse_vf_512_10x256 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:5 --data_device cuda:4 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 10 --nodes_per_layer 512 \
#--save_name isotropic_coarse_vf_512_10x512 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:6 --data_device cuda:5 &

#python -u Code/train.py --n_outputs 3 --n_dims 3 \
#--signal_file_name isotropic_coarse_vf_512.h5 \
#--n_layers 10 --nodes_per_layer 1024 \
#--save_name isotropic_coarse_vf_512_10x1024 \
#--points_per_iteration 200000 \
#--log_image false --log_gradient false \
#--device cuda:7 --data_device cuda:6

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 10 --nodes_per_layer 2048 \
--save_name isotropic_coarse_vf_512_10x2048 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:3 &

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf_512.h5 \
--n_layers 10 --nodes_per_layer 4096 \
--save_name isotropic_coarse_vf_512_10x4096 \
--points_per_iteration 200000 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:7 


###########################################################