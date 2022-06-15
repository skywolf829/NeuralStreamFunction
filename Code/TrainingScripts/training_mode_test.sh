#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/NeuralDualStreamFunction

python -u Code/train.py --model fSRN --training_mode uvw \
--data tornado.nc --n_dims 3 --n_outputs 3 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10000 --points_per_iteration 100000 \
--device mps --data_device mps