#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name cameraman_4x128

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 4 --nodes_per_layer 256 \
--save_name cameraman_4x256

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 4 --nodes_per_layer 512 \
--save_name cameraman_4x512

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 4 --nodes_per_layer 1024 \
--save_name cameraman_4x1024


##################################################

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 6 --nodes_per_layer 128 \
--save_name cameraman_6x128

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 6 --nodes_per_layer 256 \
--save_name cameraman_6x256

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 6 --nodes_per_layer 512 \
--save_name cameraman_6x512

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 6 --nodes_per_layer 1024 \
--save_name cameraman_6x1024

##############################################

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 8 --nodes_per_layer 128 \
--save_name cameraman_8x128

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 8 --nodes_per_layer 256 \
--save_name cameraman_8x256

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 8 --nodes_per_layer 512 \
--save_name cameraman_8x512

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 8 --nodes_per_layer 1024 \
--save_name cameraman_8x1024


###############################################

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 10 --nodes_per_layer 128 \
--save_name cameraman_10x128

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 10 --nodes_per_layer 256 \
--save_name cameraman_10x256

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 10 --nodes_per_layer 512 \
--save_name cameraman_10x512

python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 10 --nodes_per_layer 1024 \
--save_name cameraman_10x1024
