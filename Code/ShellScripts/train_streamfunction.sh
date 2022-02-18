#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name tornado_normal \
--normal true \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name tornado_orthogonal \
--normal false \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_orthogonal --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name tornado_binormal \
--binormal true \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF1.h5 \
--save_name synthetic_normal \
--normal true \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF1.h5 \
--save_name synthetic1_orthogonal \
--normal false \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_orthogonal --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:4 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF1.h5 \
--save_name synthetic1_binormal \
--binormal true \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:5 --data_device cuda:5 