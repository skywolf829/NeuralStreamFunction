#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

#python -u Code/train.py --n_outputs 1 --n_dims 3 \
#--signal_file_name synthetic_VF1.h5 \
#--n_layers 4 --nodes_per_layer 128 \
#--save_name synthetic_VF1_4x128 \
#--points_per_iteration 100000 \
#--fit_gradient true \
#--log_image false --log_gradient false \
#--device cuda:0 --data_device cuda:0 

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--dropout true --dropout_p 0.01 \
--signal_file_name synthetic_VF3_normal.h5 \
--n_layers 8 --nodes_per_layer 512 \
--save_name synthetic_VF3_8x512_dropout \
--points_per_iteration 100000 \
--iterations 10000 \
--fit_gradient true \
--loss l1 --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 

