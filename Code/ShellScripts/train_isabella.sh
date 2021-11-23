#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 4 --n_dims 3 \
--signal_file_name isabella13.h5 \
--n_layers 6 --nodes_per_layer 512 \
--save_name isabella_6x512 \
--points_per_iteration 100000 \
--log_image false --log_gradient false \
--loss l1occupancy \
--device cuda:0 --data_device cuda:0 
