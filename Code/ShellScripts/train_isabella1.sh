#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name isabel_4x128 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:0 &

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 4 --nodes_per_layer 256 \
--save_name isabel_4x256 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:1 &

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 4 --nodes_per_layer 512 \
--save_name isabel_4x512 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:2 &

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 4 --nodes_per_layer 1024 \
--save_name isabel_4x1024 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:3 &

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 6 --nodes_per_layer 128 \
--save_name isabel_6x128 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:4 &

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 6 --nodes_per_layer 256 \
--save_name isabel_6x256 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:5 &

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 6 --nodes_per_layer 512 \
--save_name isabel_6x512 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:6 &

python3 -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 6 --nodes_per_layer 1024 \
--save_name isabel_6x1024 \
--points_per_iteration 250000 \
--log_image false --log_gradient false \
--loss l1 \
--device cuda:0 --data_device cuda:7