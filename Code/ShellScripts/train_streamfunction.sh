#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction
python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF2.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name synth2 \
--points_per_iteration 200000 \
--iterations 10000 \
--dual_streamfunction true \
--loss magangle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 