import torch
import os
import netCDF4
from netCDF4 import Dataset
import h5py
import numpy as np
import time
from Other.utility_functions import nc_to_tensor, tensor_to_cdf
from math import pi

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    from netCDF4 import Dataset
    
    # 100th timestep
    
    a = Dataset(os.path.join(data_folder, "halfcylinder.nc"))
    print(a)
    
    u = np.array(a['u'][99])
    v = np.array(a['v'][99])
    w = np.array(a['w'][99])
    
    vf = np.stack([u,v,w])
    print(vf.shape)
    
    t = torch.tensor(vf).unsqueeze(0)
    print(t.shape)
    
    tensor_to_cdf(t, "halfcylinder_re160_ts100.nc")