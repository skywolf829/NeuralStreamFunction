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

def to_netcdf(t, location, channel_names = None):
    d = Dataset(location, 'w')

    # Setup dimensions
    d.createDimension('x')
    d.createDimension('y')
    dims = ['x', 'y']

    if(len(t.shape) == 5):
        d.createDimension('z')
        dims.append('z')

    # ['u', 'v', 'w']
    if(channel_names is None):
        ch_default = 'a'

    for i in range(t.shape[1]):
        if(channel_names is None):
            ch = ch_default
            ch_default = chr(ord(ch)+1)
        else:
            ch = channel_names[i]
        d.createVariable(ch, np.float32, dims)
        d[ch][:] = t[0,i].clone().detach().cpu().numpy()
    d.close()

if __name__ == '__main__':
    h5_files = os.listdir(data_folder)
    for filename in h5_files:
        print(filename)
        if(filename == ".DS_Store" or filename == "Seeds"):
            continue
        h = h5py.File(os.path.join(data_folder,filename), 'r')
        d = torch.tensor(np.array(h['data']), dtype=torch.float32).unsqueeze(0)
        h.close()
        print(d.shape)
        new_filename = filename.split('.')[0] + ".nc"
        to_netcdf(d, os.path.join(data_folder, new_filename))


    quit()