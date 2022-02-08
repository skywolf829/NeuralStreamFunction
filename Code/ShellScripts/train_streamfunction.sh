#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python3 -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF1.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name synth1 \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss magangle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python3 -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF1_normal.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name synth1_normal \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss magangle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 &

python3 -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF1_binormal.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name synth1_binormal \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss magangle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 