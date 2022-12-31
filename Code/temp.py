import torch
import os
import netCDF4
from netCDF4 import Dataset
import h5py
import numpy as np
import time
from Other.utility_functions import nc_to_tensor, tensor_to_cdf, make_coord_grid, curl, spatial_gradient
from math import pi
from scipy.spatial.transform import Rotation as R
from Models.options import *
from Models.models import create_model

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")



def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)*180/ pi #to degrees
    phi     =  np.arctan2(y,x)*180/ pi
    return np.stack([r,theta,phi])

def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]* pi/180 # to radian
    phi     = rthetaphi[2]* pi/180
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]

def spherical_to_quaternion(rthetaphi):
    return 0

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True), \
        vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    v = np.cross(a, b, axis=1)
    print(a.shape)
    print(b.shape)
    print(v.shape)
    print(a[...,None].shape)
    print(b[...,None].swapaxes(1,2).shape)
    c = np.matmul(a[...,None], b[...,None].swapaxes(1,2))
    print(v.shape)
    print(c.shape)
    s = np.linalg.norm(v)
    
    kmat = np.zeros([a.shape[0], 3, 3])
    kmat[:,0,0] = 0
    kmat[:,0,1] = -v[:,2]
    kmat[:,0,2] = v[:,1]
    kmat[:,1,0] = v[:,2]
    kmat[:,1,1] = 0
    kmat[:,1,2] = -v[:,0]
    kmat[:,2,0] = -v[:,1]
    kmat[:,2,1] = v[:,0]
    kmat[:,2,2] = 0
    #kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    rotation_matrix = np.expand_dims(np.eye(3), 0).repeat(a.shape[0], axis=0) + \
        kmat + np.matmul(kmat, kmat) * ((1 - c) / (s ** 2))
    #rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    print(rotation_matrix.shape)
    return rotation_matrix

def quaternion_multiply(a, b):
    
    result = np.zeros([a.shape[0], 4])
    result[:,0] = -b[:,1] * a[:,1] - b[:,2] * a[:,2] - b[:,3] * a[:,3] + b[:,0] * a[:,0]
    result[:,1] = b[:,1] * a[:,0] + b[:,2] * a[:,3] - b[:,3] * a[:,2] + b[:,0] * a[:,1]
    result[:,2] = -b[:,1] * a[:,3] + b[:,2] * a[:,0] + b[:,3] * a[:,1] + b[:,0] * a[:,2]
    result[:,3] = b[:,1] * a[:,2] - b[:,2] * a[:,1] + b[:,3] * a[:,0] + b[:,0] * a[:,3]
    
    return result

def psi_to_nc():
    
    psi1 = torch.tensor(np.load("psi1.npy")).reshape(64, 64, 64)
    psi2 = torch.tensor(np.load("psi2.npy")).reshape(64, 64, 64)
    
    grid = make_coord_grid(psi1.shape, "cpu", False, False)
    #grid = np.transpose(grid, (3, 0, 1, 2))
    grid = grid.flatten(0, 2).cpu().numpy()
    print(grid.shape)
    start_vec = np.array([[1,0,0]]).repeat(grid.shape[0], 0)
    r = rotation_matrix_from_vectors(start_vec, grid)
    rot = R.from_matrix(r)
    #print(rot.as_quat().shape)
    psi = np.zeros([r.shape[0], 4])
    print(psi1.shape)

    psi[:,0] = psi1.flatten().imag
    psi[:,1] = psi2.flatten().real
    psi[:,2] = psi2.flatten().imag
    psi[:,3] = psi1.flatten().real
    psi = quaternion_multiply(psi, rot.as_quat())
    psi_real = psi[:,0]
    psi_imag = psi[:,1]
    
    t = np.stack([psi_real.reshape(64, 64, 64), 
                     psi_imag.reshape(64, 64, 64)], axis=0)
    t = torch.tensor(t).unsqueeze(0)
    tensor_to_cdf(t, "psi.nc")
  
def delta_40_load():
    xyz_path = os.path.join(data_folder, "delta-40", "regular_xyz.obj")
    uvw_path = os.path.join(data_folder, "delta-40", "regular_uvw.obj")
    
    import pandas as pd
    # 128 x 41 x 41
    xyz = pd.read_csv(xyz_path, sep=" ", header=None).iloc[:,1:].astype(np.float32)
    uvw = pd.read_csv(uvw_path, sep=" ", header=None).iloc[:,1:].astype(np.float32)
    xyz = np.array(xyz)
    uvw = np.array(uvw)
    
    print(f"x: {xyz[:,0].min()} - {xyz[:,0].max()}")
    print(f"y: {xyz[:,1].min()} - {xyz[:,1].max()}")
    print(f"z: {xyz[:,2].min()} - {xyz[:,2].max()}")
    
    x_size = 128
    y_size = 41
    z_size = 41
    
    print(f"x_scale: {x_size / (xyz[:,0].max() - xyz[:,0].min())}")
    print(f"y_scale: {y_size / (xyz[:,1].max() - xyz[:,1].min())}")
    print(f"z_scale: {z_size / (xyz[:,2].max() - xyz[:,2].min())}")
    
    uvw = np.array(uvw).reshape(z_size, y_size, x_size, 3)
    uvw = torch.tensor(uvw).permute(3, 0, 1, 2).unsqueeze(0)
    tensor_to_cdf(uvw, "delta_40.nc")

def insidefluids():
    from netCDF4 import Dataset
    shape = [64, 64, 64]
    sx = torch.tensor(np.load("deltawing_sx.npy").reshape(shape))
    sy = torch.tensor(np.load("deltawing_sy.npy").reshape(shape))
    sz = torch.tensor(np.load("deltawing_sz.npy").reshape(shape))
    t = torch.stack([sx, sy, sz]).unsqueeze(0)
    tensor_to_cdf(t, "InsideFluids.nc", 
        channel_names=["sx", "sy", "sz"])
    
if __name__ == '__main__':

    '''
    start = np.array([76.63355624865802, 50.46789668912503, 76.38933124022662])
    end = np.array([60.202556278981916, 69.8793142881013, 72.61491798871297])
    num_points = 50
    
    for i in range(num_points):
        p = i / (num_points - 1)
        spot = start * (1-p) + end * p
        print(f"{spot[0]},{spot[1]},{spot[2]}")
    '''
    #data = nc_to_tensor(os.path.join(data_folder, "hill.nc"))
    #data = divergence(data)
    #tensor_to_cdf(data, os.path.join(data_folder, "hill_vort.nc"))
    t = nc_to_tensor(os.path.join(data_folder, "isotropic_1024^3.nc"))#[:,:,7:,:,:]
    print(t.shape)
    div = spatial_gradient(t,2,0) + spatial_gradient(t,1,1) + spatial_gradient(t,0,2)
    
    print(torch.abs(div).mean())
    #tensor_to_cdf(div, "Isotropic_divergence.nc", channel_names=["divergence"])