import os
import numpy as np
import torch
import h5py
from netCDF4 import Dataset
from math import pi, sin, atan, cos
import skimage
from torch import tensor
from utility_functions import tensor_to_cdf

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def vortex_x(x, y, z, x0, y0, z0, A=720):
    sym = 1 if z - z0 <= 0 else -1
    dist = (((x-x0)**2) + ((y-y0)**2) + ((z-z0)**2))
    num = sym *  -A * (z - z0) * (x - x0)
    denom = (2*pi) * dist * ((((x-x0)**2) + ((y-y0)**2))**0.5)
    return num / denom

def vortex_y(x, y, z, x0, y0, z0, A=720):
    sym = 1 if z - z0 <= 0 else -1
    dist = (((x-x0)**2) + ((y-y0)**2) + ((z-z0)**2))
    num = sym *  -A * (z - z0) * (y - y0)
    denom = (2*pi) * dist * ((((x-x0)**2) + ((y-y0)**2))**0.5)
    return num / denom

def vortex_z(x, y, z, x0, y0, z0, A=720):
    sym = 1 if z - z0 <= 0 else -1
    dist = (((x-x0)**2) + ((y-y0)**2) + ((z-z0)**2))
    num = sym *  A * ((((x-x0)**2)+((y-y0)**2))**0.5)
    denom = (2*pi) * dist
    return num / denom


def genereate_synthetic_vf1(resolution = 128):

    # [channels, u, v, w]
    a = np.zeros([3, resolution, resolution, resolution], dtype=np.float32)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = 1 
    end = 10
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):        
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)): 
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                u = 0.5 * (vortex_x(x, y, z, -5.5, -5.5, -5.5) + \
                    vortex_x(x, y, z, 15.0, 15.0, 15.0))
                v = 0.5 * (vortex_y(x, y, z, -5.5, -5.5, -5.5) + \
                    vortex_y(x, y, z, 15.0, 15.0, 15.0))
                w = 0.5 * (vortex_z(x, y, z, -5.5, -5.5, -5.5) + \
                    vortex_z(x, y, z, 15.0, 15.0, 15.0))
                a[:,k,j,i] = np.array([u, v, w], dtype=np.float32)
                print("%0.02f %0.02f %0.02f" % (x, y, z))
                print("%i %i %i" % (i, j, k))
                k += 1
            j += 1
        i += 1
    print(a.max())
    print(a.min())
    print(a.mean())
    print(np.linalg.norm(a, axis=0).max())
    a /= np.linalg.norm(a, axis=0).max()
    h = h5py.File("synthetic_VF1.h5", 'w')
    h['data'] = a
    h.close()

    tensor_to_cdf(torch.tensor(a).unsqueeze(0).type(torch.float32), "synthetic_VF1.cdf")

def genereate_synthetic_vf2(resolution = 128, a=2):

    # [channels, u, v, w]
    vf = np.zeros([3, resolution, resolution, resolution], dtype=np.float32)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = 1 
    end = 10
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):  
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):

                r = (((x-5)**2) + ((y-5)**2))**0.5

                u = (a**2)/(r**2) * cos(2*atan((y-5)/(x-5))) - 1
                v = (a**2)/(r**2) * sin(atan((y-5)/(x-5))*2)
                w = 0

                vf[:,k,j,i] = np.array([u, v, w], dtype=np.float32)
                k += 1
            j += 1
        i += 1
    print(vf.max())
    print(vf.min())
    print(vf.mean())
    print(np.linalg.norm(vf, axis=0).max())
    vf /= np.linalg.norm(vf, axis=0).max()
    h = h5py.File("synthetic_VF2.h5", 'w')
    h['data'] = vf
    h.close()

    tensor_to_cdf(torch.tensor(vf).unsqueeze(0).type(torch.float32), "synthetic_VF2.cdf")


if __name__ == '__main__':
    genereate_synthetic_vf1()
    #genereate_synthetic_vf2()