#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF1.h5 \
--save_name explicit_vector_reconstruction_synthetic1 \
--n_layers 4 --nodes_per_layer 128 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF2.h5 \
--save_name explicit_vector_reconstruction_synthetic2 \
--n_layers 4 --nodes_per_layer 128 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name explicit_vector_reconstruction_synthetic3 \
--n_layers 4 --nodes_per_layer 128 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name explicit_vector_reconstruction_tornado \
--n_layers 4 --nodes_per_layer 128 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name isabel.h5 \
--save_name explicit_vector_reconstruction_isabel \
--n_layers 4 --nodes_per_layer 256 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:4 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name explicit_vector_reconstruction_plume \
--n_layers 4 --nodes_per_layer 512 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:5 --data_device cuda:5