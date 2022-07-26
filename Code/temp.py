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
    
    psi1 = torch.tensor(np.load("psi1.npy"))
    psi2 = torch.tensor(np.load("psi2.npy"))
    
    t = torch.stack([psi1, psi2], dim=0).unsqueeze(0)
    
    tensor_to_cdf(t, "psi.nc")