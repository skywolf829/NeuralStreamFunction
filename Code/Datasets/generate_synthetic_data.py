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
from utility_functions import tensor_to_cdf, tensor_to_h5, jacobian, normal, binormal, spatial_gradient
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
    #r = (x**2 + y**2)**0.5
    #mask = r < a
    #theta = torch.atan(y/x)

    u = ((a**2)*(x**2 - y**2))/((x**2 + y**2)**2) - 1
    v = ((a**3)*x*y) / ((x**2 + y**2)**2)
    
    #u = torch.cos(2*theta) / r**2 - 1
    #v = torch.sin(2*theta) / r**2
    w = torch.zeros_like(u)
    
    vf = torch.stack([u, v, w], dim=0)
    #vf = vf * ~mask

    dudx = -(a**3)*x*(x**2-3*y**2) / ((x**2 + y**2)**3)
    dudy = -(a**3)*y*(y**2-3*x**2) / ((x**2 + y**2)**3)
    dudz = torch.zeros_like(dudx)
    dvdx = (a**3)*y*(y**2-3*x**2) / ((x**2 + y**2)**3)
    dvdy = (a**3)*x*(x**2-3*y**2) / ((x**2 + y**2)**3)
    dvdz = torch.zeros_like(dudx)
    dwdx = torch.zeros_like(dudx)
    dwdy = torch.zeros_like(dudx)
    dwdz = torch.zeros_like(dudx)

    J = torch.stack([
        torch.stack([dudx, dudy, dudz]),
        torch.stack([dvdx, dvdy, dvdz]),
        torch.stack([dwdx, dwdy, dwdz])
    ])
    print(J.shape)
    J = J.flatten(2).permute(2,0,1)
    print(J.shape)
    print(vf.shape)
    vf = vf.flatten(1).permute(1,0).unsqueeze(2)
    print(vf.shape)
    Jv = torch.bmm(J, vf)
    print(Jv.shape)
    b = torch.cross(Jv, vf)
    print(b.shape)
    n = torch.cross(b, vf)
    tensor_to_cdf(b.squeeze().permute(1,0).reshape(1,3,128,128,128), "binormal.nc")
    tensor_to_cdf(n.squeeze().permute(1,0).reshape(1,3,128,128,128), "normal.nc")
    #tensor_to_cdf(vf.unsqueeze(0).type(torch.float32), 
    #    "cylinder.nc")    

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
    start = -2
    end = 2
    
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

'''
Comments from Professor Crawfis's original code hosted at
http://web.cse.ohio-state.edu/~crawfis.3/Data/Tornado/tornadoSrc.c

Gen_Tornado creates a vector field of dimension [xs,ys,zs,3] from
a proceedural function. By passing in different time arguements,
a slightly different and rotating field is created.

The magnitude of the vector field is highest at some funnel shape
and values range from 0.0 to around 0.4 (I think).

I just wrote these comments, 8 years after I wrote the function.

Developed by Roger A. Crawfis, The Ohio State University
'''
def generate_crawfis_tornado(x_res, y_res, z_res, time=0):

    tornado = np.zeros([z_res, y_res, x_res, 3])
    r2 = 8
    SMALL = 0.00000000001
    xdelta = 1.0 / (x_res-1)
    ydelta = 1.0 / (y_res-1)
    zdelta = 1.0 / (z_res-1)

    z_ind = 0
    for z in np.arange(0.0, 1.0, zdelta):
        xc = 0.5 + 0.1*sin(0.04*time+10.0*z);           #For each z-slice, determine the spiral circle.
        yc = 0.5 + 0.1*cos(0.03*time+3.0*z);            #(xc,yc) determine the center of the circle.
        r = 0.1 + 0.4 * z*z + 0.1 * z * sin(8.0*z);     #The radius also changes at each z-slice.
        r2 = 0.2 + 0.1*z;                               #r is the center radius, r2 is for damping
        y_ind = 0               
        for y in np.arange(0.0, 1.0, ydelta):
            x_ind = 0
            for x in np.arange(0.0, 1.0, xdelta):
                temp = ( (y-yc)*(y-yc) + (x-xc)*(x-xc) ) ** 0.5
                scale = abs( r - temp )
                '''
                I do not like this next line. It produces a discontinuity 
                in the magnitude. Fix it later.
                '''
                if ( scale > r2 ):
                    scale = 0.8 - scale
                else:
                    scale = 1.0

                z0 = 0.1 * (0.1 - temp*z )
                if ( z0 < 0.0 ):
                    z0 = 0.0

                temp = ( temp*temp + z0*z0 )**0.5
                scale = (r + r2 - temp) * scale / (temp + SMALL)
                scale = scale / (1+z)

                # In u,v,w order 
                tornado[z_ind, y_ind, x_ind, 0] = scale * (y-yc) + 0.1*(x-xc)
                tornado[z_ind, y_ind, x_ind, 1] = scale * -(x-xc) + 0.1*(y-yc)
                tornado[z_ind, y_ind, x_ind, 2] = scale * z0

                x_ind = x_ind + 1
            y_ind = y_ind + 1
        z_ind = z_ind + 1
    
    return tornado

def generate_lorenz_attractor(x_res, y_res, z_res, sigma=10, beta=8/3, rho=28):

    vf = np.zeros([z_res, y_res, x_res, 3])
    
    start = -50
    end = 50
    xdelta = (end-start) / (x_res-1)
    ydelta = (end-start) / (y_res-1)
    zdelta = (end-start) / (z_res-1)

    z_ind = 0
    for z in np.arange(start, end, zdelta): 
        
        y_ind = 0               
        for y in np.arange(start, end, ydelta):

            x_ind = 0
            for x in np.arange(start, end, xdelta):
                
                vf[z_ind, y_ind, x_ind, 0] = sigma * (y-x)
                vf[z_ind, y_ind, x_ind, 1] = x * (rho - z) - y
                vf[z_ind, y_ind, x_ind, 2] = x*y - beta*z

                x_ind = x_ind + 1
            y_ind = y_ind + 1
        z_ind = z_ind + 1
    
    return vf


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
    #generate_flow_past_cylinder(resolution=128)
    #t = generate_crawfis_tornado(128, 128, 128, 0)
    #t = torch.tensor(t).permute(3, 0, 1, 2).unsqueeze(0).type(torch.float32)
    t = generate_lorenz_attractor(128, 128, 128)
    t = torch.tensor(t).permute(3, 0, 1, 2).unsqueeze(0).type(torch.float32)
    tensor_to_cdf(t, "lorenz.nc")
    quit()