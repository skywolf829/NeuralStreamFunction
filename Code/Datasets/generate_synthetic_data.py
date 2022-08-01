import os
import numpy as np
import torch
import h5py
from netCDF4 import Dataset
from math import pi, sin, atan, cos, tan
import skimage
import sys
from torch import tensor
script_dir = os.path.dirname(__file__)
utility_fn_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(utility_fn_dir)
from utility_functions import tensor_to_cdf, tensor_to_h5, jacobian, normal, binormal
import h5py


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
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

def generate_vortices_data(resolution = 128):

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
                #print("%0.02f %0.02f %0.02f" % (x, y, z))
                #print("%i %i %i" % (i, j, k))
                k += 1
            j += 1
        i += 1
    print(a.max())
    print(a.min())
    print(a.mean())
    print(np.linalg.norm(a, axis=0).max())
    a /= np.linalg.norm(a, axis=0).max()
    
    channel_names = ['u', 'v', 'w']
    #tensor_to_h5(torch.tensor(a).unsqueeze(0).type(torch.float32), 
    #    "vortices.h5", channel_names) 
    tensor_to_cdf(torch.tensor(a).unsqueeze(0).type(torch.float32), 
        "vortices.nc", channel_names) 

def generate_flow_past_cylinder(resolution = 128, a=2):
    start = - 5
    end = 5
    
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)
    x = zyx[2].clone()
    y = zyx[1].clone()
    z = zyx[0].clone()
    r = (x**2 + y**2)**0.5
    theta = torch.atan(y/x)

    u = torch.cos(2*theta) / r**2 - 1
    v = torch.sin(2*theta) / r**2
    w = torch.zeros_like(u)
    
    vf = torch.stack([u, v, w], dim=0)
    
    tensor_to_cdf(torch.tensor(vf).unsqueeze(0).type(torch.float32), 
        "flow_past_cylinder.nc")    
    tensor_to_h5(torch.tensor(vf).unsqueeze(0).type(torch.float32), 
        "flow_past_cylinder.h5")

def generate_ABC_flow(resolution = 128, 
                      A=np.sqrt(3), B=np.sqrt(2), C=1):
    
    start = 0
    end = 2*np.pi
    
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)
    x = zyx[2].clone()
    y = zyx[1].clone()
    z = zyx[0].clone()
    
    u = A*torch.sin(z) + C*torch.cos(y)
    v = B*torch.sin(x) + A*torch.cos(z)
    w = C*torch.sin(y) + B*torch.cos(x)
    
    abc = torch.stack([u,v,w], dim=0).unsqueeze(0)
    print(abc.shape)
    print(abc.max())
    print(abc.min())
    print(abc.mean())
    print(abc.norm(dim=1).max())
    abc /= abc.norm(dim=1).max()
    
    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(abc.type(torch.float32), 
        "ABC_flow.h5", channel_names)
    tensor_to_cdf(abc.type(torch.float32), 
        "ABC_flow.nc", channel_names)

def generate_hill_vortex(resolution = 128, 
                      A=np.sqrt(3), B=np.sqrt(2), C=1):
    start = -1
    end = 1
    
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)
    x = zyx[2].clone()
    y = zyx[1].clone()
    z = zyx[0].clone()
    r = (x**2 + y**2 + z**2)**0.5
    u = (r > 1).type(torch.LongTensor) * (3*y*x)/(2*(r**5)) + \
        (r <= 1).type(torch.LongTensor) * (1.5*y*x)
    v = (r > 1).type(torch.LongTensor) * ((3*y**2) - r**2)/(2*(r**5)) + \
        (r <= 1).type(torch.LongTensor) * (1.5*(1 - 2*(r**2)))
    w = (r > 1).type(torch.LongTensor) * (3*y*z)/(r*(r**5)) + \
        (r <= 1).type(torch.LongTensor) * (1.5*y*z)
    
    hill = torch.stack([u,v,w], dim=0).unsqueeze(0)
    print(hill.shape)
    print(hill.max())
    print(hill.min())
    print(hill.mean())
    print(hill.norm(dim=1).max())
    hill /= hill.norm(dim=1).max()
    
    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(hill.type(torch.float32), 
        "hill.nc", channel_names)

def generate_isolated_zero_vortex(resolution = 128):
    start = -1
    end = 1
    
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)
    x = zyx[2].clone()
    y = zyx[1].clone()
    z = zyx[0].clone()
    
    
    hill = torch.stack([x,y,-2*z], dim=0).unsqueeze(0)
    print(hill.shape)
    print(hill.max())
    print(hill.min())
    print(hill.mean())
    print(hill.norm(dim=1).max())
    hill /= hill.norm(dim=1).max()
    
    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(hill.type(torch.float32), 
        "isolated_zero.nc", channel_names)

def generate_non_closed_vortex(resolution = 128):
    start = -1
    end = 1
    
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)
    x = zyx[2].clone()
    y = zyx[1].clone()
    z = zyx[0].clone()
    r = (x**2 + y**2 + z**2)**0.5
    u = 1.5*(x*z) - y
    v = 1.5*(y*z) + x
    w = 1.5*(1 - 2*(x**2 + y**2) - z**2)
    
    non_closed = torch.stack([u,v,w], dim=0).unsqueeze(0)
    print(non_closed.shape)
    print(non_closed.max())
    print(non_closed.min())
    print(non_closed.mean())
    print(non_closed.norm(dim=1).max())
    non_closed /= non_closed.norm(dim=1).max()
    
    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(non_closed.type(torch.float32), 
        "non_closed.nc", channel_names)

def isabel_from_bin():
    u = np.fromfile('U.bin', dtype='>f')
    u = u.astype(np.float32)
    u[np.argwhere(u == 1e35)] = 0
    u = u.reshape([100, 500, 500])
    u = torch.tensor(u).unsqueeze(0).unsqueeze(0)

    v = np.fromfile('V.bin', dtype='>f')
    v = v.astype(np.float32)
    v[np.argwhere(v == 1e35)] = 0
    v = v.reshape([100, 500, 500])
    v = torch.tensor(v).unsqueeze(0).unsqueeze(0)

    w = np.fromfile('W.bin', dtype='>f')
    w = w.astype(np.float32)
    w[np.argwhere(w == 1e35)] = 0
    w = w.reshape([100, 500, 500])
    w = torch.tensor(w).unsqueeze(0).unsqueeze(0)

    uvw = torch.cat([u,v,w], dim=1)

    tensor_to_h5(uvw, "isabel.h5")
    tensor_to_cdf(uvw, "isabel.nc")

def plume_data_reading():
    u = np.fromfile('F:/Visualization Data/Plume/15plume3d435.ru',
                    dtype=np.float32)
    v = np.fromfile('F:/Visualization Data/Plume/15plume3d435.rv',
                    dtype=np.float32)
    w = np.fromfile('F:/Visualization Data/Plume/15plume3d435.rw',
                    dtype=np.float32)
    u = torch.tensor(u).reshape(1024, 252, 252)
    v = torch.tensor(v).reshape(1024, 252, 252)
    w = torch.tensor(w).reshape(1024, 252, 252)
    uvw = torch.stack([u, v, w]).unsqueeze(0)
    tensor_to_h5(uvw, "plume.h5")
   
def generate_vortices_seed_points():    
    seeds = torch.rand([100, 3])*2-1
    seeds *= 64
    seeds += 64
    import csv
    with open('vortices_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
        
def generate_cylinder_seed_points():    
    seeds = torch.rand([100, 3])*2-1
    seeds *= 64
    seeds += 64
    import csv
    with open('cylinder_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())

def generate_ABC_flow_seed_points():    
    seeds = torch.rand([100, 3])*2-1
    seeds *= 16
    seeds[:,0] += 32
    seeds[:,1] += 64
    seeds[:,2] += 64
    import csv
    with open('ABC_flow_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
            
def generate_tornado_seed_points():    
    seeds = torch.rand([100, 3])*2-1
    seeds[:,0] *= 16
    seeds[:,0] += 32
    seeds[:,1] *= 16
    seeds[:,1] += 32
    seeds[:,2] *= 64
    seeds[:,2] += 64
    import csv
    with open('tornado_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
   
def generate_isabel_seed_points():    
    seeds = torch.rand([200, 3])*2-1
    seeds[:100,0] *= 50
    seeds[:100,0] += 350
    seeds[:100,1] *= 50
    seeds[:100,1] += 350
    seeds[:100,2] *= 50
    seeds[:100,2] += 50
    
    seeds[100:,0] *= 80
    seeds[100:,0] += 100
    seeds[100:,1] *= 80
    seeds[100:,1] += 100
    seeds[100:,2] *= 50
    seeds[100:,2] += 50

    import csv
    with open('isabel_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
    
def generate_plume_seed_points():    
    seeds = torch.rand([200, 3])*2-1
    seeds[:50,0] *= 20
    seeds[:50,0] += 125
    seeds[:50,1] *= 20
    seeds[:50,1] += 125
    seeds[:50,2] *= 20
    seeds[:50,2] += 511
    
    seeds[50:100,0] *= 128
    seeds[50:100,0] += 128
    seeds[50:100,1] *= 128
    seeds[50:100,1] += 128
    seeds[50:100,2] *= 10
    seeds[50:100,2] += 790
    
    seeds[100:,0] *= 128
    seeds[100:,0] += 128
    seeds[100:,1] *= 128
    seeds[100:,1] += 128
    seeds[100:,2] *= 50
    seeds[100:,2] += 236

    import csv
    with open('plume_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
      
def generate_seed_files():
    generate_vortices_seed_points()
    generate_cylinder_seed_points()
    generate_ABC_flow_seed_points()
    generate_tornado_seed_points()
    generate_isabel_seed_points()
    generate_plume_seed_points()      
                       
if __name__ == '__main__':
    torch.manual_seed(0)
    #generate_seed_files()
    #generate_flow_past_cylinder(resolution=10, a=2)
    #generate_vortices_data(resolution=10)
    generate_isolated_zero_vortex(resolution=64)
    #generate_non_closed_vortex(resolution=64)
    quit()