import os
import numpy as np
import torch
import h5py
from netCDF4 import Dataset
from math import pi, sin, atan, cos, tan
import skimage
from torch import tensor
from utility_functions import tensor_to_cdf, tensor_to_h5, jacobian, normal, binormal

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

def generate_synthetic_vf1(resolution = 128):

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

def generate_synthetic_vf2(resolution = 128, a=1):

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
    h = h5py.File("synthetic_VF2.h5", 'w')
    h['data'] = vf
    h.close()
    channel_names = ['u', 'v', 'w']

    tensor_to_cdf(torch.tensor(vf).unsqueeze(0).type(torch.float32), 
        "synthetic_VF2.cdf", channel_names)

def generate_synthetic_vf2_jacobian(resolution = 128, a=1):

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
        "synthetic_VF2_jacobian.cdf", channel_names=channel_names)

def generate_synthetic_vf2_binormal(resolution = 128, device="cpu"):

    
    start = -2.5
    end = 2.5
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
    theta = torch.atan2(y, x)

    u = torch.cos(2*theta) / r**2 - 1
    v = torch.sin(2*theta) / r**2
    w = torch.zeros_like(u)

    dudx = (2*y*torch.sin(2*theta) - 2*x*torch.cos(2*theta)) \
        / (r**2)
    dudy = (-2*x*torch.sin(2*theta) - y*torch.cos(2*theta)) \
        / (r**2)
    dudz = torch.zeros_like(u)
    dvdx = (-2*x*torch.sin(2*theta) - y*torch.cos(2*theta)) \
        / (r**2)
    dvdy = (-y*torch.sin(2*theta) + 2*x*torch.cos(2*theta)) \
        / (r**2)
    dvdz = torch.zeros_like(u)
    dwdx = torch.zeros_like(u)
    dwdy = torch.zeros_like(u)
    dwdz = torch.zeros_like(u)

    vf = torch.stack([u,v,w], dim=0)
    jac = torch.stack(
            [torch.stack([dudx, dudy, dudz], dim=0), 
            torch.stack([dvdx, dvdy, dvdz], dim=0), 
            torch.stack([dwdx, dwdy, dwdz], dim=0)],
            dim=0)
    

    jac2 = jacobian(vf.unsqueeze(0))
    b2 = binormal(vf.unsqueeze(0), jac2)
    n2 = normal(vf.unsqueeze(0), b=b2)
 
    vf_max_norm = vf.norm(dim=3).max()
    vf /= vf_max_norm
    
    Jt = torch.bmm(jac.flatten(2).permute(2, 0, 1) / vf_max_norm, 
        vf.flatten(1).unsqueeze(1).permute(2, 0, 1))

    vf_binorm = torch.cross(Jt, 
        vf.flatten(1).unsqueeze(1).permute(2, 0, 1))
    vf_norm = torch.cross(vf_binorm, 
        vf.flatten(1).unsqueeze(1).permute(2, 0, 1))


    print(vf.shape)
    print(vf_binorm.shape)
    print(vf_norm.shape)
    vf_binorm = vf_binorm.reshape(
        [resolution, resolution, resolution, 3, 1]
        ).squeeze().permute(3, 0, 1, 2)
    vf_norm = vf_norm.reshape(
        [resolution, resolution, resolution, 3, 1]
        ).squeeze().permute(3, 0, 1, 2)
    
    print(vf.shape)
    print(vf_binorm.shape)
    print(vf_norm.shape)
    print(vf_norm.norm(dim=0).shape)
    vf_norm /= vf_norm.norm(dim=0)
    vf_norm *= vf.norm(dim=0)
    vf_binorm /= vf_binorm.norm(dim=0)
    vf_binorm *= vf.norm(dim=0)

    print(jac.shape)
    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(vf_norm.unsqueeze(0), 
        "synthetic_VF2_normal.cdf", channel_names=channel_names)
        
    tensor_to_cdf(n2, 
        "synthetic_VF2_normal2.cdf", channel_names=channel_names)
    tensor_to_cdf(vf_binorm.unsqueeze(0), 
        "synthetic_VF2_binormal.cdf", channel_names=channel_names)

        
    tensor_to_cdf(b2, 
        "synthetic_VF2_binormal2.cdf", channel_names=channel_names)
    tensor_to_cdf(vf.unsqueeze(0), 
        "synthetic_VF2.cdf", channel_names=channel_names)

    tensor_to_cdf(jac.flatten(0,1).unsqueeze(0), 
        "synthetic_VF2_jacobian.cdf")
    tensor_to_cdf(jac2[0].flatten(0,1).unsqueeze(0), 
        "synthetic_VF2_jacobian2.cdf")

    tensor_to_h5(vf.unsqueeze(0).type(torch.float32), 
        "synthetic_VF2.h5")  
    tensor_to_h5(vf_norm.unsqueeze(0).type(torch.float32), 
        "synthetic_VF2_normal.h5")
    tensor_to_h5(vf_binorm.unsqueeze(0).type(torch.float32), 
        "synthetic_VF2_binormal.h5")

def generate_synthetic_vf3(resolution = 128, A=np.sqrt(3), B=np.sqrt(2), C=1):
    # [channels, u, v, w]
    a = np.zeros([3, resolution, resolution, resolution], dtype=np.float32)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = 0
    end = 2*np.pi
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):        
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)): 
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                u = A*np.sin(z) + C*np.cos(y)
                v = B*np.sin(x) + A*np.cos(z)
                w = C*np.sin(y) + B*np.cos(x)
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
    h = h5py.File("synthetic_VF3.h5", 'w')
    h['data'] = a
    h.close()
    channel_names = ['u', 'v', 'w']

    tensor_to_cdf(torch.tensor(a).unsqueeze(0).type(torch.float32), 
        "synthetic_VF3.cdf", channel_names)

def generate_synthetic_vf3_jacobian(resolution = 128, A=np.sqrt(3), B=np.sqrt(2), C=1):

    # [channels, u, v, w]
    vf = np.zeros([9, resolution, resolution, resolution], dtype=np.float32)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = 0
    end = 2*np.pi
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):  
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                dudx = 0
                dudy = -C*np.sin(y)
                dudz = A*np.cos(z)
                dvdx = B*np.cos(x)
                dvdy = 0
                dvdz = -A*np.sin(z)
                dwdx = -B*np.sin(x)
                dwdy = C*np.cos(y)
                dwdz = 0

                vf[:,k,j,i] = np.array([dudx, dudy, dudz, 
                    dvdx, dvdy, dvdz, dwdx, dwdy, dwdz], dtype=np.float32)
                k += 1
                print("%f %f %f" % (x, y, z))
            j += 1
        i += 1
    
    channel_names = ['dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz', 'dwdx', 'dwdy', 'dwdz']
    tensor_to_cdf(torch.tensor(vf).unsqueeze(0).type(torch.float32) / np.max(np.abs(vf)), 
        "synthetic_VF3_jacobian.cdf", channel_names=channel_names)

def generate_synthetic_vf3_binormal(resolution = 128, A=np.sqrt(3), B=np.sqrt(2), C=1, device="cpu"):

    # [channels, u, v, w]
    jac = torch.zeros([resolution, resolution, resolution, 3, 3], 
    dtype=torch.float32, device=device)
    vf = torch.zeros([resolution, resolution, resolution, 3, 1], 
    dtype=torch.float32, device=device)

    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = 0
    end = 2*np.pi
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):  
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                dudx = 0
                dudy = -C*np.sin(y)
                dudz = A*np.cos(z)
                dvdx = B*np.cos(x)
                dvdy = 0
                dvdz = -A*np.sin(z)
                dwdx = -B*np.sin(x)
                dwdy = C*np.cos(y)
                dwdz = 0

                u = A*np.sin(z) + C*np.cos(y)
                v = B*np.sin(x) + A*np.cos(z)
                w = C*np.sin(y) + B*np.cos(x)

                vf[k,j,i,:, 0] = torch.tensor([u, v, w], 
                    dtype=torch.float32, device=device)

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

def generate_synthetic_vf4(resolution = 128):
    torch.seed(0)

    f = torch.randn([1, 1, 16, 16, 16])
    g = torch.randn([1, 3, 16, 16, 16])
    import torch.nn.functional as F

    f = F.interpolate(f, [resolution, resolution, resolution])
    g = torch.randn([1, 3, 16, 16, 16])

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

if __name__ == '__main__':
    # u*iHat + v*jHat + w*kHat
    #genereate_synthetic_vf1()
    generate_synthetic_vf2_binormal()
    #generate_synthetic_vf3()
    
    quit()