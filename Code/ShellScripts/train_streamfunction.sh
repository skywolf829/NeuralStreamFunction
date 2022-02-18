#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name isabel.h5 \
--save_name isabel_normal_4x128 \
--normal true \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name isabel.h5 \
--save_name isabel_normal_4x512 \
--normal true \
--n_layers 4 --nodes_per_layer 512 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 
