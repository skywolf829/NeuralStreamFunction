import torch
import os
import netCDF4
from netCDF4 import Dataset
import h5py
import numpy as np
import time
from Other.utility_functions import nc_to_tensor

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    test_file = "isotropic_1024^3.nc"
    t = time.time()
    data = Dataset(os.path.join(data_folder, test_file))
    t_1 = time.time()
    print(f"Took {t_1-t : 0.04f} sec. to load dataset as NetCDF")
    t = time.time()
    channels = []
    for a in data.variables:
        d = np.array(data[a])
        channels.append(d)
    t_1 = time.time()
    d = np.stack(channels)
    print(d.shape)
    print(f"Took {t_1-t : 0.04f} sec. to load dataset to RAM (numpy).")
    t = time.time()
    d = torch.tensor(d).unsqueeze(0)
    t_1 = time.time()
    print(d.shape)
    print(f"Took {t_1-t : 0.04f} sec. to move numpy data to tensor format.")
    quit()