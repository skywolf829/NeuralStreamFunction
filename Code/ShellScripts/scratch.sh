#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name test \
--n_layers 4 --nodes_per_layer 128 \
--residual false \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient false --gradient_direction N \
--dual_stream_function any \
--loss angle_parallel \
--lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0