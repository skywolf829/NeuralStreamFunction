#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_parallel_large \
--n_layers 8 --nodes_per_layer 512 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_same_large \
--n_layers 8 --nodes_per_layer 512 \
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
--signal_file_name plume.h5 \
--save_name plume_parallel_small \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_same_small \
--n_layers 6 --nodes_per_layer 256 \
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
--signal_file_name plume.h5 \
--save_name plume_parallel_med \
--n_layers 6 --nodes_per_layer 1024 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:4 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_same_med \
--n_layers 6 --nodes_per_layer 1024 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:5 --data_device cuda:5 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_parallel_xl \
--n_layers 10 --nodes_per_layer 1024 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:6 --data_device cuda:6 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name plume_same_xl \
--n_layers 10 --nodes_per_layer 1024 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:7 --data_device cuda:7 