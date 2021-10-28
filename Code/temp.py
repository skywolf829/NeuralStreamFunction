import os
import numpy as np
import torch
import h5py
from netCDF4 import Dataset


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    f_h5 = h5py.File(os.path.join(data_folder, "isotropic1024coarse_ts1.h5"), 'r')
    d = torch.tensor(f_h5['data'])

    print(d.max())
    print(d.min())