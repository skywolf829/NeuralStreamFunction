#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution
export PATH="$HOME/sz/bin:$PATH"
#export PATH="$HOME/zfp/bin:$PATH"
#export PATH="$HOME/fpzip/bin:$PATH"
export PATH="$HOME/tthresh/build:$PATH"

# Isomag2D
#python3 -u Code/mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool2D \
#--upscaling_technique model --model_name Isomag2D --criterion psnr --start_metric 27 \
#--end_metric 60 --metric_skip 0.5 --output_folder Isomag2D_datareduction --max_LOD 6 \
#--min_chunk 16 --mode 2D --file Isomag2D.h5 --dims 2 --nx 1024 --ny 1024 \
#--use_compressor true --distributed false --compressor sz --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u Code/TestingScripts/sz_test.py --file Isomag2D.h5 \
#--dims 2 --nx 1024 --ny 1024 --output_folder Isomag2D_datareduction \
#--start_value 10 --end_value 60 --value_skip 1 --metric psnr \
#--save_netcdf false --device cuda:0

# 3D iso1024 mag
python3 -u Code/mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D \
--upscaling_technique model --model_name Isomag3D --criterion psnr --start_metric 30 \
--end_metric 60 --metric_skip 1.5 --output_folder Isomag3D_datareduction --max_LOD 4 \
--min_chunk 32 --mode 3D --file Isomag3D.h5 --dims 3 --nx 1024 --ny 1024 --nz 1024 \
--use_compressor true --distributed true --compressor sz --load_existing false \
--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR false \
--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

python3 -u Code/mixedLOD_octree.py --save_name NN_TTHRESH --downscaling_technique avgpool3D \
--upscaling_technique model --model_name Isomag3D --criterion psnr --start_metric 30 \
--end_metric 60 --metric_skip 1.5 --output_folder Isomag3D_datareduction --max_LOD 4 \
--min_chunk 32 --mode 3D --file Isomag3D.h5 --dims 3 --nx 1024 --ny 1024 --nz 1024 \
--use_compressor true --distributed true --compressor tthresh --load_existing false \
--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR false \
--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

python3 -u Code/TestingScripts/sz_test.py --file Isomag3D.h5 \
--dims 3 --nx 1024 --ny 1024 --nz 1024 --output_folder Isomag3D_datareduction \
--start_value 10 --end_value 60 --value_skip 1 --metric psnr \
--save_netcdf false --device cuda:0

python3 -u Code/TestingScripts/tthresh_test.py --file Isomag3D.h5 \
--dims 3 --nx 1024 --ny 1024 --nz 1024 --output_folder Isomag3D_datareduction \
--start_value 10 --end_value 60 --value_skip 1 --metric psnr \
--save_netcdf false --device cuda:0

# 3D mixing dataset
#python3 -u Code/mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool2D \
#--upscaling_technique model --model_name Mixing3D --criterion psnr --start_metric 25 \
#--end_metric 60 --metric_skip 0.5 --output_folder Mixing3D_datareduction_sz --max_LOD 4 \
#--min_chunk 16 --mode 3D --file Mixing3D.h5 --dims 3 --nx 512 --ny 512 --nz 512 \
#--use_compressor true --distributed false --compressor sz --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u Code/mixedLOD_octree.py --save_name NN_TTHRESH --downscaling_technique avgpool2D \
#--upscaling_technique model --model_name Mixing3D --criterion psnr --start_metric 25 \
#--end_metric 60 --metric_skip 0.5 --output_folder Mixing3D_datareduction_sz --max_LOD 4 \
#--min_chunk 16 --mode 3D --file Mixing3D.h5 --dims 3 --nx 512 --ny 512 --nz 512 \
#--use_compressor true --distributed false --compressor sz --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 


# Plume dataset
#python3 -u Code/mixedLOD_octree.py --save_name NN_TTHRESH --downscaling_technique avgpool3D \
#--upscaling_technique model --model_name Plume --criterion psnr --start_metric 25 \
#--end_metric 60 --metric_skip 1.0 --output_folder Plume_datareduction --max_LOD 3 \
#--min_chunk 16 --mode 3D --file Plume.h5 --dims 3 --nx 512 --ny 512 --nz 128 \
#--use_compressor true --distributed false --compressor tthresh --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u Code/mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D \
#--upscaling_technique model --model_name Plume --criterion psnr --start_metric 25 \
#--end_metric 60 --metric_skip 1.0 --output_folder Plume_datareduction --max_LOD 3 \
#--min_chunk 16 --mode 3D --file Plume.h5 --dims 3 --nx 512 --ny 512 --nz 128 \
#--use_compressor true --distributed false --compressor sz --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u Code/TestingScripts/sz_test.py --file Plume.h5 \
#--dims 3 --nx 512 --ny 128 --nz 128 --output_folder Plume_datareduction \
#--start_value 10 --end_value 60 --value_skip 1 --metric psnr \
#--save_netcdf false --device cuda:0

#python3 -u Code/TestingScripts/tthresh_test.py --file Plume.h5 \
#--dims 3 --nx 512 --ny 128 --nz 128 --output_folder Plume_datareduction \
#--start_value 10 --end_value 60 --value_skip 1 --metric psnr \
#--save_netcdf false --device cuda:0

# Vorts dataset
#python3 -u Code/mixedLOD_octree.py --save_name NN_TTHRESH --downscaling_technique avgpool3D \
#--upscaling_technique model --model_name Vorts --criterion psnr --start_metric 25 \
#--end_metric 60 --metric_skip 1.0 --output_folder Vorts_datareduction --max_LOD 3 \
#--min_chunk 16 --mode 3D --file Vorts.h5 --dims 3 --nx 128 --ny 128 --nz 128 \
#--use_compressor true --distributed false --compressor tthresh --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u Code/mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D \
#--upscaling_technique model --model_name Vorts --criterion psnr --start_metric 25 \
#--end_metric 60 --metric_skip 1.0 --output_folder Vorts_datareduction --max_LOD 3 \
#--min_chunk 16 --mode 3D --file Vorts.h5 --dims 3 --nx 128 --ny 128 --nz 128 \
#--use_compressor true --distributed false --compressor sz --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u Code/TestingScripts/sz_test.py --file Vorts.h5 \
#--dims 3 --nx 128 --ny 128 --nz 128 --output_folder Vorts_datareduction \
#--start_value 10 --end_value 60 --value_skip 1 --metric psnr \
#--save_netcdf false --device cuda:0

#python3 -u Code/TestingScripts/tthresh_test.py --file Vorts.h5 \
#--dims 3 --nx 128 --ny 128 --nz 128 --output_folder Vorts_datareduction \
#--start_value 10 --end_value 60 --value_skip 1 --metric psnr \
#--save_netcdf false --device cuda:0


