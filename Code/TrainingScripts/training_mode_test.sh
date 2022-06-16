#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/NeuralDualStreamFunction

python -u Code/train.py --model fSRN --training_mode uvw \
--data tornado.nc --n_dims 3 --n_outputs 3 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_uvw \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode uvwf_any \
--data tornado.nc --n_dims 3 --n_outputs 4 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_uvwf_any \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode uvwf_parallel \
--data tornado.nc --n_dims 3 --n_outputs 4 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_uvwf_parallel \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode uvwf_direction \
--data tornado.nc --n_dims 3 --n_outputs 4 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_uvwf_direction \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode dsf_any \
--data tornado.nc --n_dims 3 --n_outputs 2 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_dsf_any \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode dsf_parallel \
--data tornado.nc --n_dims 3 --n_outputs 2 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_dsf_parallel \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode dsf_direction \
--data tornado.nc --n_dims 3 --n_outputs 2 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_dsf_direction \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode dsfm_any \
--data tornado.nc --n_dims 3 --n_outputs 3 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_dsfm_any \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode dsfm_parallel \
--data tornado.nc --n_dims 3 --n_outputs 3 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_dsfm_parallel \
--device cuda --data_device cuda

python -u Code/train.py --model fSRN --training_mode dsfm_direction \
--data tornado.nc --n_dims 3 --n_outputs 3 \
--n_layers 4 --nodes_per_layer 256 \
--iterations 10 --points_per_iteration 100000 \
--save_name tornado_dsfm_direction \
--device cuda --data_device cuda