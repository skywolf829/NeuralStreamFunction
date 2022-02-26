#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--signal_file_name isotropic_coarse_vf.h5 \
--n_layers 8 --nodes_per_layer 512 \
--save_name isotropic_coarse_perpendicular_noconstraint \
--points_per_iteration 1000000 \
--loss perpendicular \
--log_image true --log_gradient false \
--device cuda:0 --data_device cuda:0 
