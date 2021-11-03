#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution


python -u Code/train.py --n_outputs 1 --n_dims 2 \
--vector_field_name cameraman.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name cameraman_4x128_262144periter \
--points_per_iteration 262144
