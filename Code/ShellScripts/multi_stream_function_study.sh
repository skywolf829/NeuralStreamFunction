#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/NeuralDualStreamFunction

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name multi_stream_function_study_tornado_N \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--gradient_direction N \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name multi_stream_function_study_tornado_N_parallel \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--gradient_direction N \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name multi_stream_function_study_tornado_any \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--gradient_direction V \
--loss angle_orthogonal --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name multi_stream_function_study_tornado_dsf_N \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--dual_stream_function N \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name multi_stream_function_study_tornado_dsf_N_parallel \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--dual_stream_function N \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:4 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name multi_stream_function_study_tornado_dsf_any \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--dual_stream_function any \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:5 --data_device cuda:5