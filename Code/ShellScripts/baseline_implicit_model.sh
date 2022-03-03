#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isabel.h5 \
--save_name baseline_implict_model_isabel_lg \
--n_layers 6 --nodes_per_layer 512 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction false \
--normal false \
--norm true \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name tornado3d.h5 \
--save_name baseline_implict_model_tornado_lg \
--n_layers 6 --nodes_per_layer 512 \
--residual false \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction false \
--normal false \
--norm true \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 