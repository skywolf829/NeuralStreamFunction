#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--vector_field_name isotropic_coarse_mag.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name isomag_4x128 \
--points_per_iteration 500000
