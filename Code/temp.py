import os
import numpy as np
import torch
import h5py
from netCDF4 import Dataset
import skimage

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    f_h5 = h5py.File(os.path.join(data_folder, "cameraman.h5"), 'w')
    
    a = torch.tensor(skimage.data.camera(), dtype=torch.float32).unsqueeze(0) / 255
    print(a.max())
    print(a.min())
    f_h5['data'] = a.cpu()
    f_h5.close()