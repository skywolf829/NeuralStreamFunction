from __future__ import absolute_import, division, print_function
import argparse
import datetime
import torch
import time
import os
import numba as nb
import numpy as np
from typing import Tuple, List
import sys
script_dir = os.path.dirname(__file__)
utility_fn_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(utility_fn_dir)
from utility_functions import jacobian, normal, nc_to_tensor, tensor_to_cdf

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

@nb.njit
def clamp(val, lower, upper):
    to_return = val
    if(val < lower):
        to_return = lower
    elif(val > upper):
        to_return = upper
    return to_return

@nb.njit
def trilinear_interpolate(vol : np.ndarray, 
    x : float, y : float, z : float) -> np.ndarray:
    
    z = clamp(z, 0, vol.shape[2]-1)
    y = clamp(y, 0, vol.shape[3]-1)
    x = clamp(x, 0, vol.shape[4]-1)
    
    x0 : int = int(x)
    x1 : int = x0 + 1
    
    y0 : int = int(y)
    y1 : int = y0 + 1

    z0 : int = int(z)
    z1 : int = z0 + 1
    
    z0 = clamp(z0, 0, vol.shape[2]-1)
    y0 = clamp(y0, 0, vol.shape[3]-1)
    x0 = clamp(x0, 0, vol.shape[4]-1)
    
    z1 = clamp(z1, 0, vol.shape[2]-1)
    y1 = clamp(y1, 0, vol.shape[3]-1)
    x1 = clamp(x1, 0, vol.shape[4]-1)
    
    x1_diff = x1-x
    x0_diff = x-x0    
    y1_diff = y1-y
    y0_diff = y-y0
    z1_diff = z1-z
    z0_diff = z-z0
    
    c00 = vol[0,:,z0,y0,x0] * z1_diff + vol[0,:,z1,y0,x0]*z0_diff
    c01 = vol[0,:,z0,y0,x1] * z1_diff + vol[0,:,z1,y0,x1]*z0_diff
    c10 = vol[0,:,z0,y1,x0] * z1_diff + vol[0,:,z1,y1,x0]*z0_diff
    c11 = vol[0,:,z0,y1,x1] * z1_diff + vol[0,:,z1,y1,x1]*z0_diff

    c0 = c00 * y1_diff + c10 * y0_diff
    c1 = c01 * y1_diff + c11 * y0_diff

    c = c0 * x1_diff + c1 * x0_diff
    c = c.astype(vol.dtype)
    return c

@nb.njit()
def previous_point(i : int, j : int, k : int) -> np.ndarray:
    to_return : np.ndarray = np.array([0,0,0])
    if(i == 0 and j == 0 and k != 0):
        to_return[0] = 0
        to_return[1] = 0
        to_return[2] = k-1
    elif(i == 0 and j != 0):
        to_return[0] = 0
        to_return[1] = j-1
        to_return[2] = k
    else:
        to_return[0] = i-1
        to_return[1] = j
        to_return[2] = k
    return to_return


@nb.njit()
def princpal_stream_function(
    sf : np.ndarray,
    vf : np.ndarray,
    n : np.ndarray):
    
    dt = np.array([0.5], dtype=np.float32)
    eps = np.array([1e-10], dtype=np.float32)

    vf_shape = np.array(list(vf.shape[2:]), dtype=np.float32) 
    for k in range(0, vf.shape[2]):
        for j in range(0, vf.shape[3]):
            for i in range(0, vf.shape[4]):
                
                
                if i == 0 and j == 0 and k == 0:
                    sf[k,j,i] = 0
                else:
                    pp = previous_point(i,j,k)
                    p = np.array([i,j,k], dtype=np.float32)

                    V = vf[0,:,pp[2],pp[1],pp[0]]

                    V_plus_spot = p + V * dt            
                    V_plus = trilinear_interpolate(vf, 
                            V_plus_spot[0],
                            V_plus_spot[1],
                            V_plus_spot[2])    
                    
                    V_minus_spot = p - V * dt
                    V_minus = trilinear_interpolate(vf, 
                            V_minus_spot[0],
                            V_minus_spot[1],
                            V_minus_spot[2])
                    
                    P_plus = p + ((V + V_plus)/2.0) * dt
                    P_minus = p - ((V + V_minus)/2.0) * dt
                    
                    dL = (V * dt + ((V_plus + V_minus)/2.0) * dt)
                    A = (trilinear_interpolate(vf,
                            P_plus[0],
                            P_plus[1],
                            P_plus[2]
                        ) - trilinear_interpolate(vf,
                            P_minus[0],
                            P_minus[1],
                            P_minus[2])) 
                    A /= (dL+eps)
                    B = np.cross(V, A)
                    B /= (np.linalg.norm(B)+eps)
                    N = np.cross(B, V) * np.linalg.norm(V)
                    dp = (p - pp) / (vf_shape-1)
                    
                    
                    dp_step = 1 / (vf_shape[0]-1)
                    sf[k,j,i] = \
                        sf[pp[2], pp[1], pp[0]] + \
                            n[0,0,pp[2],pp[1],pp[0]]*dp[0]+n[0,1,pp[2],pp[1],pp[0]]*dp[1]+n[0,2,pp[2],pp[1],pp[0]]*dp[2]# + \
                                #0.5 * (A[0]*dp[0]**2 + A[1]*dp[1]**2 + A[2]*dp[2]**2 )
                    #sf[i,j,k] = \
                    #    sf[pp[2], pp[1], pp[0]] + \
                    #        np.linalg.norm(V) * dp_step + \
                    #            0.5 * np.linalg.norm(A) * dp_step**2

    sf -= sf.min()
    sf /= sf.max()
    return sf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device',default=None, type=str,
        help='Which device to use')
    parser.add_argument('--data_device',default=None,type=str,
        help='Which device to keep the data on')
    parser.add_argument('--data',default=None,type=str,
        help='Data file name')

    args = vars(parser.parse_args())

    torch.manual_seed(11235813)

    now = datetime.datetime.now()
    start_time = time.time()    
    
    print(f"Loading and preprocessing the normal and jacobian fields for the vector field.")
    
    vf = nc_to_tensor(os.path.join(data_folder, args['data']))    
    print(vf.max())
    print(vf.min())
    shape : List[int] = [vf.shape[2], vf.shape[3], vf.shape[4]]
    sf : np.ndarray = np.zeros(shape, np.float32)
    n = normal(vf, normalize=False)
    n /= n.norm(dim=1, keepdim=True)
    n *= vf.norm(dim=1, keepdim=True)

    # Fix for the cylinder dataset
    #mask = n[0,1] > 0
    #mask = mask.unsqueeze(0).unsqueeze(0)
    #n = n*mask + (-n* ~mask)
    

    tensor_to_cdf(n, "cylinder_normal.nc")
    end_time = time.time()
    print(f"Finished preprocessing in {end_time-start_time : 0.02f} seconds")
    print(f"Normal vf shape {n.shape}")
    print(f"Beginning principal stream function calculation.")
    start_time = time.time()

    sf = princpal_stream_function(
        sf.astype(np.float32),
        vf.cpu().numpy().astype(np.float32),
        n.cpu().numpy().astype(np.float32))

    end_time = time.time()
    print(f"Finished stream function calculation in {end_time-start_time : 0.02f} seconds")
    
    tensor_to_cdf(torch.tensor(sf).unsqueeze(0).unsqueeze(0), "sf_"+args['data'])