#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python3 -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name synth3_dual_streamfunction \
--n_layers 4 --nodes_per_layer 256 \
--points_per_iteration 100000 \
--iterations 10000 \
--dual_streamfunction true \
--loss l1 --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 