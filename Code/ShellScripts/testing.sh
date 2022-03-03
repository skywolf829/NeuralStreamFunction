#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name plume.h5 \
--save_name explicit_vector_reconstruction_plume_test \
--n_layers 8 --nodes_per_layer 300 \
--residual true \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0