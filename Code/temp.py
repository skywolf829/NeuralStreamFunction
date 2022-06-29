import torch
import os
import netCDF4
from netCDF4 import Dataset
import h5py
import numpy as np
import time
from Other.utility_functions import nc_to_tensor
from math import pi

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    center = np.array([40, 64, 70])
    r = 10
    
    f = open("tornado_seeding_curve2.csv", 'w')
    lines = []
    for i in np.linspace(0, 2*pi, 50):
        x = center[0]+r*np.cos(i)
        y = center[1]
        z = center[2]+r*np.sin(i)
        print(f"{x} {y} {z}")
        lines.append(str(x) + ","+str(y)+","+str(z)+"\n")
    f.writelines(lines)
    quit()