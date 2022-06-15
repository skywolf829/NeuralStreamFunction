import torch
import os
import netCDF4
from netCDF4 import Dataset
import h5py
import numpy as np


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    test_file = "tornado.nc"
    d = Dataset(os.path.join(data_folder, test_file))
    channels = []
    for a in d.variables:
        n = np.array(d[a])
        channels.append(n)
    n = np.stack(channels)
    n = torch.tensor(n).unsqueeze(0)
    print(n.shape)
    quit()