#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name vortices.h5 \
--save_name loss_function_test_vortices_orthogonal \
--n_layers 4 --nodes_per_layer 128 \
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
--signal_file_name vortices.h5 \
--save_name loss_function_test_vortices_parallel \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name vortices.h5 \
--save_name loss_function_test_vortices_same \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name vortices.h5 \
--save_name loss_function_test_vortices_eq \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name loss_function_test_tornado_orthogonal \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_orthogonal --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:4 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name loss_function_test_tornado_parallel \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:5 --data_device cuda:5 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name loss_function_test_tornado_same \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:6 --data_device cuda:6 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name loss_function_test_tornado_eq \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:7 --data_device cuda:7 
