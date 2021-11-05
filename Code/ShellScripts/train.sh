#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name cat_4x128 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:0 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 4 --nodes_per_layer 256 \
--save_name cat_4x256 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:1 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 4 --nodes_per_layer 512 \
--save_name cat_4x512 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:2 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 4 --nodes_per_layer 1024 \
--save_name cat_4x1024 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:3 &


###########################################################

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 6 --nodes_per_layer 128 \
--save_name cat_6x128 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:4 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 6 --nodes_per_layer 256 \
--save_name cat_6x256 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:5 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 6 --nodes_per_layer 512 \
--save_name cat_6x512 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:6 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 6 --nodes_per_layer 1024 \
--save_name cat_6x1024 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:7


###########################################################

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 8 --nodes_per_layer 128 \
--save_name cat_8x128 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:0 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 8 --nodes_per_layer 256 \
--save_name cat_8x256 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:1 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 8 --nodes_per_layer 512 \
--save_name cat_8x512 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:2 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 8 --nodes_per_layer 1024 \
--save_name cat_8x1024 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:3 &


###########################################################

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 10 --nodes_per_layer 128 \
--save_name cat_10x128 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:4 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 10 --nodes_per_layer 256 \
--save_name cat_10x256 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:5 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 10 --nodes_per_layer 512 \
--save_name cat_10x512 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:6 &

python -u Code/train.py --n_outputs 3 --n_dims 2 \
--signal_file_name cat.h5 \
--n_layers 10 --nodes_per_layer 1024 \
--save_name cat_10x1024 \
--points_per_iteration 200000 \
--log_image true --log_gradient true \
--device cuda:7


###########################################################