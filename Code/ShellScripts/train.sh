#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF2.h5 \
--save_name synthetic2_test \
--n_layers 4 --nodes_per_layer 128 \
--points_per_iteration 100000 \
--iterations 1000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 