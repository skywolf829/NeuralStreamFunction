#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_N_parallel \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--binormal false \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:0 --data_device cuda:0 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_B_parallel \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal false \
--binormal true \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_dsf_parallel \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--binormal false \
--norm false \
--norm_per_voxel false \
--loss angle_parallel --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:2 --data_device cuda:2 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_N_same \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--binormal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:3 --data_device cuda:3 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_B_same \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal false \
--binormal true \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:4 --data_device cuda:4 &

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_dsf_same \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--binormal false \
--norm false \
--norm_per_voxel false \
--loss angle_same --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:5 --data_device cuda:5 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_N_eq \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal true \
--binormal false \
--norm false \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:6 --data_device cuda:6 &

python -u Code/train.py --n_outputs 1 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_B_eq \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient true \
--dual_streamfunction false \
--normal false \
--binormal true \
--norm false \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:7 --data_device cuda:7 

python -u Code/train.py --n_outputs 2 --n_dims 3 \
--signal_file_name synthetic_VF3.h5 \
--save_name multi_streamfunction_study_synthetic3_dsf_eq \
--n_layers 6 --nodes_per_layer 256 \
--points_per_iteration 200000 \
--iterations 10000 \
--fit_gradient false \
--dual_streamfunction true \
--normal false \
--binormal false \
--norm false \
--norm_per_voxel false \
--loss mse --lr 5e-5 \
--log_image false --log_gradient false \
--device cuda:1 --data_device cuda:1 