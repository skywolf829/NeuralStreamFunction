import os
import numpy as np
import torch
import h5py
from netCDF4 import Dataset
from math import pi, sin, atan, cos, tan
import skimage
from torch import tensor
from utility_functions import tensor_to_cdf, tensor_to_h5

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
    channel_names = ['u', 'v', 'w']

    tensor_to_cdf(torch.tensor(a).unsqueeze(0).type(torch.float32), 
        "synthetic_VF1.cdf", channel_names)

def genereate_synthetic_vf2(resolution = 128, a=1):

    # [channels, u, v, w]
    vf = np.zeros([3, resolution, resolution, resolution], dtype=np.float32)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = -2.5
    end = 2.5
    center = (end+start)/2
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):  
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                y1 = y-center
                x1 = x-center
                r = (x1**2 + y1**2)**0.5
                theta = atan(y1/x1)
                u = (a**2)/(r**2) * cos(2*theta) - 1
                v = (a**2)/(r**2) * sin(2*theta)
                w = 0

                vf[:,k,j,i] = np.array([u, v, w], dtype=np.float32)
                k += 1
                print("%f %f %f" % (x, y, z))
            j += 1
        i += 1
    print(vf.max())
    print(vf.min())
    print(vf.mean())
    print(np.linalg.norm(vf, axis=0).max())
    vf /= np.linalg.norm(vf, axis=0).max()
    print(vf.max())
    print(vf.min())
    print(vf.mean())
    h = h5py.File("synthetic_VF3.h5", 'w')
    h['data'] = vf
    h.close()
    channel_names = ['u', 'v', 'w']

    tensor_to_cdf(torch.tensor(vf).unsqueeze(0).type(torch.float32), 
        "synthetic_VF3.cdf", channel_names)

def genereate_synthetic_vf2_jacobian(resolution = 128, a=1):

    # [channels, u, v, w]
    vf = np.zeros([9, resolution, resolution, resolution], dtype=np.float32)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = -2.5
    end = 2.5
    center = (end+start)/2
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):  
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                y1 = y-center
                x1 = x-center
                r = (x1**2 + y1**2)**0.5
                theta = atan(y1/x1)
                dudx = (2*a*y1*sin(2*atan(theta)) - a*x*cos(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dudy = (-2*a*x1*sin(2*atan(theta)) + a*y*cos(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dudz = 0
                dvdx = (-a*x1*sin(2*atan(theta)) - 2*a*y*cos(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dvdy = (2*a*x1*cos(2*atan(theta)) - a*y*sin(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dvdz = 0
                dwdx = 0
                dwdy = 0
                dwdz = 0

                vf[:,k,j,i] = np.array([dudx, dudy, dudz, 
                    dvdx, dvdy, dvdz, dwdx, dwdy, dwdz], dtype=np.float32)
                k += 1
                print("%f %f %f" % (x, y, z))
            j += 1
        i += 1
    print(vf.max())
    print(vf.min())
    print(vf.mean())
    #print(np.linalg.norm(vf, axis=0).max())
    #vf /= np.linalg.norm(vf, axis=0).max()
    #h = h5py.File("synthetic_VF3_jacobian.h5", 'w')
    #h['data'] = vf
    #h.close()
    channel_names = ['dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz', 'dwdx', 'dwdy', 'dwdz']
    tensor_to_cdf(torch.tensor(vf).unsqueeze(0).type(torch.float32) / np.max(np.abs(vf)), 
        "synthetic_VF3_jacobian.cdf", channel_names=channel_names)

def genereate_synthetic_vf2_binormal(resolution = 128, a=1, device="cpu"):

    # [channels, u, v, w]
    jac = torch.zeros([resolution, resolution, resolution, 3, 3], 
    dtype=torch.float32, device=device)
    vf = torch.zeros([resolution, resolution, resolution, 3, 1], 
    dtype=torch.float32, device=device)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = -2.5
    end = 2.5
    center = (end+start)/2
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):  
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                y1 = y-center
                x1 = x-center
                r = (x1**2 + y1**2)**0.5
                theta = atan(y1/x1)
                dudx = (2*a*y1*sin(2*atan(theta)) - a*x*cos(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dudy = (-2*a*x1*sin(2*atan(theta)) + a*y*cos(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dudz = 0
                dvdx = (-a*x1*sin(2*atan(theta)) - 2*a*y*cos(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dvdy = (2*a*x1*cos(2*atan(theta)) - a*y*sin(2*atan(theta))) \
                    / (((x1**2)+(y1**2))**1.5)
                dvdz = 0
                dwdx = 0
                dwdy = 0
                dwdz = 0

                u = (a**2)/(r**2) * cos(2*theta) - 1
                v = (a**2)/(r**2) * sin(2*theta)
                w = 0

                vf[k,j,i,:, 0] = torch.tensor([u, v, w], 
                    dtype=torch.float32, device=device)

                #jac[k,j,i,:,:] = torch.tensor(
                #    [[dudx, dudy, dudz], 
                #    [dvdx, dvdy, dvdz], 
                #    [dwdx, dwdy, dwdz]], 
                #    dtype=torch.float32, device=device)
                jac[k,j,i,:,:] = torch.tensor(
                    [[dudx, dvdx, dwdx], 
                    [dudy, dvdy, dwdy], 
                    [dudz, dvdz, dwdz]], 
                    dtype=torch.float32, device=device)
                k += 1
                print("%f %f %f" % (x, y, z))
            j += 1
        i += 1
    vf_max_norm = vf.norm(dim=3).max()
    vf /= vf_max_norm
    
    Jt = torch.bmm(jac.flatten(0,2) / vf_max_norm, 
        vf.flatten(0,2))

    print(Jt.shape)
    vf_binorm = torch.cross(Jt, vf.flatten(0,2))
    vf_norm = torch.cross(vf_binorm, vf.flatten(0,2))

    print(vf_norm.shape)
    print( vf_norm.norm(dim=1).unsqueeze(-1).shape)
    print(vf.flatten(0,2).norm(dim=1).shape)
    vf_norm = vf_norm / vf_norm.norm(dim=1).unsqueeze(-1)
    vf_norm *= vf.flatten(0,2).norm(dim=1).unsqueeze(-1)
    
    vf_binorm = vf_binorm / vf_binorm.norm(dim=1).unsqueeze(-1)
    vf_binorm *= vf.flatten(0,2).norm(dim=1).unsqueeze(-1)

    vf_binorm = vf_binorm.reshape([resolution, resolution, resolution, 3]).permute(3, 0, 1, 2)
    vf_norm = vf_norm.reshape([resolution, resolution, resolution, 3]).permute(3, 0, 1, 2)
    
    print(vf_norm.norm(dim=0).max())
    print(vf_binorm.norm(dim=0).max())

    vf_norm /= vf_norm.norm(dim=0).max()
    vf_binorm /= vf_binorm.norm(dim=0).max()

    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(vf_norm.unsqueeze(0).type(torch.float32), 
        "synthetic_VF3_normal.cdf", channel_names=channel_names)
    tensor_to_cdf(vf_binorm.unsqueeze(0).type(torch.float32), 
        "synthetic_VF3_binormal.cdf", channel_names=channel_names)

        
    tensor_to_h5(vf_norm.unsqueeze(0).type(torch.float32), 
        "synthetic_VF3_normal.h5")
    tensor_to_h5(vf_binorm.unsqueeze(0).type(torch.float32), 
        "synthetic_VF3_binormal.h5")


if __name__ == '__main__':
    # u*iHat + v*jHat + w*kHat
    #genereate_synthetic_vf1()
    genereate_synthetic_vf2_binormal()
    quit()