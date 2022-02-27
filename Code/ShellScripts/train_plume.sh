#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_orthogonal_md_single \
--n_layers 6 --nodes_per_layer 512 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_orthogonal --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_same_md_single \
--n_layers 6 --nodes_per_layer 512 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_eq_md_single \
--n_layers 6 --nodes_per_layer 512 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_parallel_md_single \
--n_layers 6 --nodes_per_layer 512 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3
