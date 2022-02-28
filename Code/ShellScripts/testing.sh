#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name network_size_study_tornado_deep_residual \
--n_layers 10 --nodes_per_layer 64 \
--residual true \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3 