#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py --n_outputs 3 --n_dims 3 \
--vector_field_name isotropic_coarse_vf.h5 \
--n_layers 4 --nodes_per_layer 128 \
--save_name isovf_4x128 \
--points_per_iteration 500000
