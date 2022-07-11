from __future__ import absolute_import, division, print_function
import argparse
import datetime
from .. import Other
import torch
import time
import os
from Models.options import *
import numba as nb
import numpy as np
from typing import Tuple

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

@nb.njit()
def pp(i : int, j : int, k : int) -> Tuple[int]:
    if(i == 0 and j == 0 and k == 0):
        return 0
    elif(i == 0 and j == 0 and k != 0):
        return (0, 0, k-1)
    elif(i == 0 and j != 0):
        return (0, j-1, k)
    else:
        return (i-1, j, k)
    

@nb.njit()
def princpal_stream_function(vf : np.ndarray, 
    vf_normal : np.ndarray,
    jac : np.ndarray):
    sf = np.zeros([vf.shape[2], vf.shape[3], vf.shape[4]])
    dv = 1 / vf.shape[2]
    for k in range(0, vf.shape[4]):
        for j in range(0, vf.shape[3]):
            for i in range(0, vf.shape[2]):
                if i == 0 and j == 0 and k == 0:
                    sf[i,j,k] = 0
                else:
                    prev_point = pp(i,j,k)
                    f_prime_pp = np.linalg.norm(vf[0,:,
                        prev_point[0],prev_point[1],prev_point[2]])
                    f_prime_prime_pp = np.linalg.norm(
                        jac[0,:,prev_point[0], prev_point[1], prev_point[2]]
                    )
                    sf[i,j,k] = \
                        sf[prev_point[0], prev_point[1], prev_point[2]] + \
                            f_prime_pp * dv + \
                                f_prime_prime_pp * 0.5 * dv * dv


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
    n = normal(vf)
    j = jacobian(vf)
    end_time = time.time()
    print(f"Finished preprocessing in {end_time-start_time : 0.02f} seconds")
    print(f"Normal vf shape {n.shape}")
    print(f"Jacobian shape {j.shape}")
    print(f"Beginning principal stream function calculation.")
    start_time = time.time
    sf = princpal_stream_function(vf.cpu().numpy(), 
        n.cpu().numpy(),
        j.cpu().numpy())

    end_time = time.time()
    print(f"Finished stream function calculation in {end_time-start_time : 0.02f} seconds")