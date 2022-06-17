from __future__ import absolute_import, division, print_function
import argparse
from turtle import forward
from Datasets.datasets import Dataset
import datetime
from Other.utility_functions import str2bool, PSNR, make_coord_grid, tensor_to_cdf, ssim3D, \
    tensor_to_h5, create_folder, normal, binormal, cdf_to_tensor, get_vtr
from Models.models import load_model, save_model, ImplicitModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import h5py
import netCDF4
from Models.options import *
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import numpy as np
from vtk import vtkXMLPolyDataReader
from vtkmodules.util import numpy_support
from vtkmodules.util.numpy_support import vtk_to_numpy
from pandas import read_csv
import vtk

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--load_from',default=None,type=str)
    parser.add_argument('--supersample',default=None,type=float)
    parser.add_argument('--supersample_psnr',default=None,type=str)
    parser.add_argument('--supersample_gradient',default=None,type=float)
    parser.add_argument('--normal_field',default=None,type=str2bool)
    parser.add_argument('--implicit_jacobian',default=None,type=str2bool)
    parser.add_argument('--jacobian_discrete',default=None,type=str2bool)
    parser.add_argument('--cdf',default=None,type=str2bool)    
    parser.add_argument('--uvw',default=None,type=str2bool)
    parser.add_argument('--cdf_cross',default=None,type=str2bool)
    parser.add_argument('--uncertainty',default=None,type=str2bool)
    parser.add_argument('--grad_cdf',default=None,type=str2bool)
    parser.add_argument('--dual_streamfunction',default=None,type=str2bool)
    parser.add_argument('--seeding_curve',default=None,type=str2bool)
    parser.add_argument('--cai_method',default=None,type=str2bool)
    parser.add_argument('--explicit_recon',default=None,type=str2bool)
    parser.add_argument('--explicit_recon_uncertainty',default=None,type=str2bool)
    parser.add_argument('--hausdorff_distance',default=None,type=str2bool)
    parser.add_argument('--streamlines',default=None,type=str2bool)
    parser.add_argument('--boxplots',default=None,type=str2bool)

    parser.add_argument('--decompose',default=None,type=str2bool)
    parser.add_argument('--device',default="cuda:0",type=str)

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")


    
    if(args['hausdorff_distance'] is None and
       args['streamlines'] is None and
       args['boxplots'] is None):     
        opt = load_options(os.path.join(save_folder, args["load_from"]))
        opt["device"] = args["device"]
        opt['data_device'] = args['device']
        opt["save_name"] = args["load_from"]
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        dataset = Dataset(opt)
        model = load_model(opt, opt['device'])
        model = model.to(opt['device'])

        model.eval()

    #print(dataset.data.min())
    #print(dataset.data.mean())
    #print(dataset.data.max())

    #writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))

    if(args['supersample_psnr'] is not None):
        original_volume = h5py.File(os.path.join(data_folder, args['supersample_psnr']), 'r')['data']
        original_volume = torch.tensor(original_volume).to(opt['device']).unsqueeze(0)
        print(original_volume.shape)
        grid = list(original_volume.shape[2:])
        with torch.no_grad():
            supersampled_volume = model.sample_grid(grid)
            if(len(grid) == 3):
                supersampled_volume = supersampled_volume.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                supersampled_volume = supersampled_volume.permute(2, 0, 1).unsqueeze(0)
            tensor_to_cdf(supersampled_volume, os.path.join(output_folder, opt['save_name']+"_supersampled.cdf"))


            p_model = PSNR(original_volume, supersampled_volume, 
                range=original_volume.max()-original_volume.min()).item()
            s_model = ssim3D(original_volume, supersampled_volume).item()
            print("Neural network supersampling PSNR/SSIM: %0.03f/%0.05f" % (p_model, s_model))

            ss_interp_volume = model.sample_grid(list(dataset.data.shape[2:]))
            if(len(grid) == 3):
                ss_interp_volume = ss_interp_volume.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                ss_interp_volume = ss_interp_volume.permute(2, 0, 1).unsqueeze(0)
            ss_interp_volume = F.interpolate(ss_interp_volume, size=original_volume.shape[2:],
                align_corners=False, mode='trilinear' if len(original_volume.shape) == 5 else 'bilinear')

            tensor_to_cdf(ss_interp_volume, os.path.join(output_folder, opt['save_name']+"_with_interpolation.cdf"))
            p_ss_interp = PSNR(original_volume, ss_interp_volume,
                range=original_volume.max()-original_volume.min()).item()
            s_ss_interp = ssim3D(original_volume, ss_interp_volume).item()
            print("Network + interp supersampling PSNR/SSIM: %0.03f/%0.05f" % (p_ss_interp, s_ss_interp))

            interpolated_volume = F.interpolate(dataset.data.to(opt['device']), size=original_volume.shape[2:],
                align_corners=False, mode='trilinear' if len(original_volume.shape) == 5 else 'bilinear')

            tensor_to_cdf(interpolated_volume, os.path.join(output_folder, opt['save_name']+"_interpolated.cdf"))
            p_interp = PSNR(original_volume, interpolated_volume,
                range=original_volume.max()-original_volume.min()).item()
            s_interp = ssim3D(original_volume, interpolated_volume).item()
            print("Interpolation supersampling PSNR/SSIM: %0.03f/%0.05f" % (p_interp, s_interp))
            
            tensor_to_cdf(original_volume, os.path.join(output_folder, args['supersample_psnr']+'.cdf'))

    if(args['supersample'] is not None):
        grid = list(dataset.data.shape[2:])
        for i in range(len(grid)):
            grid[i] *= args['supersample']
            grid[i] = int(grid[i])
        with torch.no_grad():
            img = model.sample_grid(grid)
        print(img.min())
        print(img.mean())
        print(img.max())
        print(img.shape)
        writer.add_image('Supersample x'+str(args['supersample']), 
            img.clamp(dataset.min(), dataset.max()), 0, dataformats='WHC')
        
    if(args['supersample_gradient'] is not None):
        grid = list(dataset.data.shape[2:])
        for i in range(len(grid)):
            grid[i] *= args['supersample_gradient']
            grid[i] = int(grid[i])
        
        grad_img = model.sample_grad_grid(grid)
        
        for output_index in range(len(grad_img)):
            for input_index in range(grad_img[output_index].shape[-1]):
                grad_img[output_index][...,input_index] -= \
                    grad_img[output_index][...,input_index].min()
                grad_img[output_index][...,input_index] /= \
                    grad_img[output_index][...,input_index].max()
                                                
                writer.add_image('Supersample gradient_outputdim'+str(output_index)+\
                    "_wrt_inpudim_"+str(input_index), 
                    grad_img[output_index][...,input_index:input_index+1].clamp(0, 1), 
                    0, dataformats='WHC')
    
    if(args['normal_field'] is not None):
        grid = list(dataset.data.shape[2:])
        
        with torch.no_grad():
            vector_field = model.sample_grid(grid)
            print(vector_field.shape)

        jacobian = model.sample_grad_grid(grid)
        
        with torch.no_grad():
            jacobian = torch.cat(jacobian, dim=1)
            print(jacobian.shape)
            
            normal_field = torch.matmul(jacobian, vector_field)
            print(normal_field.shape)

    if(args['implicit_jacobian'] is not None):
        grid = list(dataset.data.shape[2:])
        coord_grid = make_coord_grid(grid, model.opt['device'], False)
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)       

        output_shape = list(coord_grid.shape)
        output_shape[-1] = model.opt['n_dims']
        print("Output shape")
        print(output_shape)
        output_grad = torch.empty(output_shape, 
            dtype=torch.float32, device=model.opt['device'], 
            requires_grad=False)

        output_shape.append(3)

        output_jacobian = torch.empty(output_shape, 
            dtype=torch.float32, device=model.opt['device'], 
            requires_grad=False)

        max_points = 1000
        for start in range(0, coord_grid.shape[0], max_points):
            model.zero_grad()
            points = coord_grid[start:min(start+max_points, coord_grid.shape[0])].clone()
            vals = model.net(
                points)
            grad = torch.autograd.grad(outputs=vals, 
                inputs=points, 
                grad_outputs=torch.ones_like(vals),
                create_graph=True)[0]

            output_grad[start:min(start+max_points, coord_grid.shape[0])] = grad.clone().detach()
                
            for dim in range(grad.shape[1]):
                model.zero_grad()
                grad2 = torch.autograd.grad(outputs=grad[:,dim:dim+1],
                    inputs=points,
                    grad_outputs=torch.ones_like(grad[:,dim:dim+1]),
                    retain_graph=True)[0]
                
                output_jacobian[start:min(start+max_points, coord_grid.shape[0]), dim] = \
                    grad2.detach()
                
            #print(grad2[0][start:min(start+max_points, coord_grid.shape[0])].shape)

        print(output_grad.shape)
        print(output_jacobian.shape)
        print(output_jacobian.max())
        print(output_jacobian.min())
        n = torch.bmm(output_jacobian, output_grad.unsqueeze(-1))[...,0]
        n /= torch.norm(n, dim=1).max()
        print(n.max())
        print(n.min())
        b = torch.cross(n, output_grad, dim=1)
        b /= torch.norm(b, dim=1).max()
        print(b.max())
        print(b.min())
        print(n.shape)
        print(b.shape)


        reconstructed_vf = output_grad.reshape(coord_grid_shape)
        if(len(grid) == 3):
            reconstructed_vf = reconstructed_vf.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            reconstructed_vf = reconstructed_vf.permute(2, 0, 1).unsqueeze(0)

        reconstructed_bvf = b.reshape(coord_grid_shape)
        reconstructed_nvf = n.reshape(coord_grid_shape)
        if(len(grid) == 3):
            reconstructed_bvf = reconstructed_bvf.permute(3, 0, 1, 2).unsqueeze(0)
            reconstructed_nvf = reconstructed_nvf.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            reconstructed_bvf = reconstructed_bvf.permute(2, 0, 1).unsqueeze(0)
            reconstructed_nvf = reconstructed_nvf.permute(3, 0, 1, 2).unsqueeze(0)
        
        tensor_to_cdf(reconstructed_bvf, 
            os.path.join(output_folder, opt['save_name']+"_binormal_reconstructed.cdf"),
            ['u', 'v', 'w'])

        tensor_to_cdf(reconstructed_nvf, 
            os.path.join(output_folder, opt['save_name']+"_normal_reconstructed.cdf"),
            ['u', 'v', 'w'])

    if(args['jacobian_discrete'] is not None):

        reconstructed_vf = dataset.data.clone()
        s = list(reconstructed_vf.shape)
        print("reconstructed vf shape")
        print(reconstructed_vf.shape)

        z_kernel = torch.zeros([3, 3, 3], device=opt['device'])
        z_kernel[0, 1, 1] = -0.5
        z_kernel[2, 1, 1] = 0.5
        z_kernel = z_kernel.unsqueeze(0).expand(3, 1, 3, 3, 3)
        y_kernel = torch.zeros([3, 3, 3], device=opt['device'])
        y_kernel[1, 0, 1] = -0.5
        y_kernel[1, 2, 1] = 0.5
        y_kernel = y_kernel.unsqueeze(0).expand(3, 1, 3, 3, 3)
        x_kernel = torch.zeros([3, 3, 3], device=opt['device'])
        x_kernel[1, 1, 0] = -0.5
        x_kernel[1, 1, 2] = 0.5
        x_kernel = x_kernel.unsqueeze(0).expand(3, 1, 3, 3, 3)

        j_shape = list(reconstructed_vf.shape)
        j_shape.insert(1, 3)
        

        output_jacobian  = torch.zeros(j_shape, device=opt['device'])
        
        output_jacobian[0,:,0] = F.conv3d(F.pad(reconstructed_vf, 
            mode='replicate', pad=[1, 1, 1, 1, 1, 1]),
            x_kernel, groups=reconstructed_vf.shape[1])
        output_jacobian[0,:,1] = F.conv3d(F.pad(reconstructed_vf, 
            mode='replicate', pad=[1, 1, 1, 1, 1, 1]), 
            y_kernel, groups=reconstructed_vf.shape[1])
        output_jacobian[0,:,2] = F.conv3d(F.pad(reconstructed_vf, 
            mode='replicate', pad=[1, 1, 1, 1, 1, 1]), 
            z_kernel, groups=reconstructed_vf.shape[1])

        output_jacobian_s = list(output_jacobian.shape)
        output_jacobian = output_jacobian.flatten(1,2)
        print(output_jacobian.shape)
        tensor_to_cdf(output_jacobian / output_jacobian.abs().max(), 
            os.path.join(output_folder, opt['save_name']+"_cd_jacobian.cdf"),
            ['dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz', 'dwdx', 'dwdy', 'dwdz'])
        print(output_jacobian.min())
        print(output_jacobian.max())
        print("rest")
        output_jacobian = output_jacobian.reshape(output_jacobian_s)

        output_jacobian = output_jacobian[0].permute(2, 3, 4, 0, 1).flatten(0, 2)
        print(output_jacobian.shape)
        reconstructed_vf = reconstructed_vf[0].permute(1, 2, 3, 0).flatten(0, 2)
        print(reconstructed_vf.shape)

        Jt = torch.bmm(output_jacobian, reconstructed_vf.unsqueeze(-1))[...,0]
        b = torch.cross(Jt, reconstructed_vf, dim=1)
        n = torch.cross(b, reconstructed_vf, dim=1)
        print(b.max())
        print(b.min())
        print(n.max())
        print(n.min())

        n /= n.norm(dim=1).max()
        b /= b.norm(dim=1).max()
        print(b.max())
        print(b.min())
        print(n.max())
        print(n.min())

        reconstructed_bvf = b.permute(1, 0).reshape(s)
        reconstructed_nvf = n.permute(1, 0).reshape(s)
        
        tensor_to_cdf(reconstructed_bvf, 
            os.path.join(output_folder, opt['save_name']+"_binormal_discrete.cdf"),
            ['u', 'v', 'w'])

        tensor_to_h5(reconstructed_bvf, 
            os.path.join(output_folder, opt['save_name']+"_binormal_discrete.h5"))

        tensor_to_cdf(reconstructed_nvf, 
            os.path.join(output_folder, opt['save_name']+"_normal_discrete.cdf"),
            ['u', 'v', 'w'])
            
        tensor_to_h5(reconstructed_nvf, 
            os.path.join(output_folder, opt['save_name']+"_normal_discrete.h5"))

    if(args['uvw'] is not None):
        grid = list(dataset.data.shape[2:])
        with torch.no_grad():
            reconstructed_volume = model.sample_grid(grid)
        if(len(grid) == 3):
            reconstructed_volume = reconstructed_volume.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            reconstructed_volume = reconstructed_volume.permute(2, 0, 1).unsqueeze(0)
        
        p_ss_interp = PSNR(dataset.data, reconstructed_volume,
            range=dataset.max()-dataset.min()).item()
        #s_ss_interp = ssim3D(dataset.data, reconstructed_volume).item()
        #s_ss_interp = 1.0
        print("Model %s - Reconstructed PSNR/SSIM: %0.03f/%0.05f" % \
            (opt['save_name'], p_ss_interp, 0))
        cos_dist = F.cosine_similarity(dataset.data,#,
            reconstructed_volume, dim=1)
        print(f"Cosine dist stats - min/mean/max {cos_dist.min().item() : 0.04f}/{cos_dist.mean().item() : 0.04f}/{cos_dist.max().item() : 0.04f}")
        print(f"Avg/std err - {(1-cos_dist).abs().mean().item() : 0.04f}/{cos_dist.std().item() : 0.04f}")

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        counts, bins = np.histogram(
            cos_dist.flatten().detach().cpu().numpy(), 
            bins=100,
            range=(-1.0, 1.0))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts, 
                label=f"Avg error:  {(1-cos_dist.abs()).abs().mean().item() : 0.04f}")
        #plt.title("Cosine similarity between network gradient and vector field")
        plt.ylabel("Proportion")
        plt.xlabel("Cosine similarity")
        plt.legend()
        plt.show()
        create_folder(output_folder, opt['save_name'])
        tensor_to_cdf(reconstructed_volume, 
            os.path.join(output_folder, opt['save_name'], "reconstructed.cdf"))
        tensor_to_cdf(dataset.data, 
            os.path.join(output_folder, opt['save_name'], "original.cdf"))

    if(args['cdf'] is not None):
        grid = list(dataset.data.shape[2:])
        with torch.no_grad():
            t1 = time.time()
            reconstructed_volume = model.sample_grid(grid)
            torch.cuda.synchronize()
            elapsed = time.time() - t1
            print(f"Computation took {elapsed : 0.05f}")
            total_num_verts = dataset.data.shape[2]*dataset.data.shape[3]*dataset.data.shape[4]
        if(len(grid) == 3):
            reconstructed_volume = reconstructed_volume.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            reconstructed_volume = reconstructed_volume.permute(2, 0, 1).unsqueeze(0)
        
        #p_ss_interp = PSNR(dataset.data, reconstructed_volume,
        #    range=dataset.max()-dataset.min()).item()
        #s_ss_interp = ssim3D(dataset.data, reconstructed_volume).item()
        #s_ss_interp = 1.0
        #print("Model %s - Reconstructed PSNR/SSIM: %0.03f/%0.05f" % \
        #    (opt['save_name'], p_ss_interp, s_ss_interp))
        create_folder(output_folder, opt['save_name'])
        tensor_to_cdf(reconstructed_volume, 
            os.path.join(output_folder, 
                         opt['save_name'], "reconstructed.nc"))
        #tensor_to_cdf(dataset.data, 
        #    os.path.join(output_folder, opt['save_name'], "original.nc"))

    if(args['cdf_cross'] is not None):
        x = make_coord_grid(dataset.data.shape[2:], 
                    opt['data_device'], flatten=True).unsqueeze(0)
        for _ in range(len(dataset.data.shape[2:])-1):
            x = x.unsqueeze(-2)
        y = F.grid_sample(dataset.data, 
            x, mode='nearest', align_corners=False)
        x = x.squeeze()
        y = y.squeeze()
        if(len(y.shape) == 1):
            y = y.unsqueeze(0)        
        y = y.permute(1,0)
        reconstructed_volume = model.forward_w_grad(x)
        y_estimated, x = model.forward_w_grad(x)

        grads_f = torch.autograd.grad(y_estimated[:,0], x, 
                grad_outputs=torch.ones_like(y_estimated[:,0]),
                create_graph=True)[0].detach()
        grads_g = torch.autograd.grad(y_estimated[:,1], x, 
                grad_outputs=torch.ones_like(y_estimated[:,1]),
                create_graph=True)[0].detach()
        y_estimated = torch.cross(grads_f, grads_g, dim=1)

        p = PSNR(y, y_estimated,
            range=dataset.max()-dataset.min()).item()
    
    if(args['uncertainty'] is not None):
        grid = list(dataset.data.shape[2:])
        model.train(True)
        forward_passes = []
        n_passes = 500
        for i in range(n_passes):
            with torch.no_grad():
                reconstructed_volume = model.sample_grid(grid)
            if(len(grid) == 3):
                reconstructed_volume = reconstructed_volume.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                reconstructed_volume = reconstructed_volume.permute(2, 0, 1).unsqueeze(0)
            
            if(i == 0):
                s = list(reconstructed_volume.shape)
                s[0] = n_passes
                forward_passes = torch.zeros(s, device=opt['device'])
            forward_passes[i] = reconstructed_volume.detach()
            print(reconstructed_volume[0,0,50,50,50])
        forward_passes = forward_passes.to("cpu")
        v = torch.var(forward_passes, dim=0, keepdim=True).to(opt['device'])
        m = torch.mean(forward_passes, dim=0, keepdim=True).to(opt['device'])
        del forward_passes
        
        #p_ss_interp = PSNR(dataset.data, m,
        #    range=dataset.max()-dataset.min()).item()
        #s_ss_interp = ssim3D(dataset.data, reconstructed_volume).item()
        #s_ss_interp = 1.0
        #print("Model %s - Reconstructed PSNR/SSIM: %0.03f/%0.05f" % \
        #    (opt['save_name'], p_ss_interp, s_ss_interp))
        print(m.shape)
        print(v.shape)
        tensor_to_cdf(torch.cat([m, v], dim=1), 
            os.path.join(output_folder, opt['save_name']+"_uncertainty.cdf"),
            ['mean', 'variance'])
        #tensor_to_cdf(torch.abs(m-dataset.data), 
        #    os.path.join(output_folder, opt['save_name']+"_error.cdf"))

        model.train(False)
        
    if(args['grad_cdf'] is not None):
        grid = list(dataset.data.shape[2:])
        reconstructed_volume = model.sample_grad_grid(grid)
        if(len(grid) == 3):
            reconstructed_volume = reconstructed_volume.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            reconstructed_volume = reconstructed_volume.permute(2, 0, 1).unsqueeze(0)
        folder_to_load = os.path.join(data_folder, 
            opt['signal_file_name'])
        f = h5py.File(folder_to_load, 'r')
        d = torch.tensor(
            np.array(f.get('data'))
            ).unsqueeze(0).to(opt['data_device'])
        f.close()
        if(opt['norm']):
            d /= (d.norm(dim=1) + 1e-8)
        elif(opt['norm_per_voxel']):
            d /= (d.norm(dim=1).max() + 1e-8)

        cos_dist = F.cosine_similarity(d,#dataset.data,#,
            reconstructed_volume, dim=1)
        print(f"Cosine dist stats - min/mean/max {cos_dist.min().item() : 0.04f}/{cos_dist.mean().item() : 0.04f}/{cos_dist.max().item() : 0.04f}")
        print(f"Avg/std err - {(cos_dist).abs().mean().item() : 0.04f}/{cos_dist.std().item() : 0.04f}")

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        counts, bins = np.histogram(
            cos_dist.flatten().detach().cpu().numpy(), 
            bins=100,
            range=(-1.0, 1.0))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts, 
                label=f"Avg error:  {(cos_dist.abs()).abs().mean().item() : 0.04f}")
        #plt.title("Cosine similarity between network gradient and vector field")
        plt.ylabel("Proportion")
        plt.xlabel("Cosine similarity")
        plt.legend()
        plt.show()
        #p_ss_interp = PSNR(dataset.data, reconstructed_volume,
            #range=dataset.max()-dataset.min()).item()
        #s_ss_interp = ssim3D(dataset.data, reconstructed_volume).item()
        #print("Reconstructed PSNR/SSIM: %0.03f/%0.05f" % (p_ss_interp, s_ss_interp))
        create_folder(output_folder, opt['save_name'])
        tensor_to_cdf(reconstructed_volume.detach(), 
           os.path.join(output_folder, opt['save_name'], "grad_reconstructed.nc"))
        tensor_to_cdf(cos_dist.detach().unsqueeze(0), 
            os.path.join(output_folder, opt['save_name'], "cos_dist.nc"))
     
    if(args['explicit_recon'] is not None):
        grid = list(dataset.data.shape[2:])
        
        print("Sampling grad 1")
        grads_f = model.sample_grad_grid(grid, output_dim=0, 
                                        max_points=10000)
        print("Sampling grad 2")
        grads_g = model.sample_grad_grid(grid, output_dim=1, 
                                        max_points=10000)
        
        grads_f = grads_f.permute(3, 0, 1, 2).unsqueeze(0)
        grads_g = grads_g.permute(3, 0, 1, 2).unsqueeze(0)
        
        recon = torch.cross(grads_f, grads_g, dim=1)
        recon /= (recon.norm(dim=1) + 1e-8)
        
        d = dataset.data
        d /= (d.norm(dim=1) + 1e-8)
        err = F.cosine_similarity(recon, d)
        
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        counts, bins = np.histogram(
            err.flatten().detach().cpu().numpy(), 
            bins=100,
            range=(-1.0, 1.0))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts,
                label=f"Avg error:  {(1-err.abs()).abs().mean().item() : 0.06f}")
        plt.title("Cos. sim. between V and network cross product")
        plt.ylabel("Proportion")
        plt.xlabel("Cosine similarity")
        plt.legend()
        plt.show()
        
        create_folder(output_folder, opt['save_name'])
        #recon /= recon.norm(dim=1)
        tensor_to_cdf(recon.detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "dual_streamfunction.nc"))
        
        tensor_to_cdf(err.unsqueeze(0).detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "cross_product_cos_dist.nc"))
        tensor_to_cdf(d.detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "original.nc"))
        
    if(args['explicit_recon_uncertainty'] is not None):
        grid = list(dataset.data.shape[2:])
        
        model.train(False)
        forward_passes = []
        n_passes = 10
        
        for i in range(1):
            print("Sampling grad 1")
            grads_f = model.sample_grad_grid(grid, output_dim=0, 
                                            max_points=10000)
            print("Sampling grad 2")
            grads_g = model.sample_grad_grid(grid, output_dim=1, 
                                            max_points=10000)
            
            grads_f = grads_f.permute(3, 0, 1, 2).unsqueeze(0)
            grads_g = grads_g.permute(3, 0, 1, 2).unsqueeze(0)
            
            recon = torch.cross(grads_f, grads_g, dim=1)
            forward_passes.append(recon)
        forward_passes = torch.cat(forward_passes)
        forward_passes = forward_passes.to("cpu")
        v = torch.var(forward_passes, dim=0, keepdim=True).to(opt['device'])
        recon = torch.mean(forward_passes, dim=0, keepdim=True).to(opt['device'])
        del forward_passes  
        
        err = F.cosine_similarity(recon, dataset.data)
        
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        counts, bins = np.histogram(
            err.flatten().detach().cpu().numpy(), 
            bins=100,
            range=(-1.0, 1.0))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts,
                label=f"Avg error:  {(1-err.abs()).abs().mean().item() : 0.04f}")
        plt.title("Cos. sim. between V and network cross product")
        plt.ylabel("Proportion")
        plt.xlabel("Cosine similarity")
        plt.legend()
        plt.show()
        
        model.train(True)
        fgs = []
        for i in range(n_passes):
            print("Sampling grid")
            with torch.no_grad():
                fg = model.sample_grid(grid)  
                fgs.append(fg)
        fgs = torch.stack(fgs)
        fgs = fgs.to("cpu")
        fg_v = torch.var(fgs, dim=0).to(opt['device'])
        fg = torch.mean(fgs, dim=0).to(opt['device'])
        del fgs  
        
        print(fg_v.shape)
        print(fg.shape)
        fg_v = fg_v.permute(3, 0, 1, 2).unsqueeze(0).sum(dim=1, keepdim=True)
        fg = fg.permute(3, 0, 1, 2).unsqueeze(0)
        print(fg_v.shape)
        print(fg.shape)
        create_folder(output_folder, opt['save_name'])
        
        tensor_to_cdf(recon.detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "dual_streamfunction.nc"))
        
        tensor_to_cdf(err.unsqueeze(0).detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "cross_product_cos_dist.nc"))
        
        tensor_to_cdf(fg.detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "reconstructed.nc"))
        tensor_to_cdf(fg_v.detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "variance.nc"))

    if(args['dual_streamfunction'] is not None):
        grid = list(dataset.data.shape[2:])
        print("Sampling grid")
        with torch.no_grad():
            y_estimated = model.sample_grid(grid)

        print("Sampling grad 1")
        grads_f = model.sample_grad_grid(grid, output_dim=0,
                                        max_points=10000)
        print("Sampling grad 2")
        grads_g = model.sample_grad_grid(grid, output_dim=1,
                                        max_points=10000)
        
        grads_f = grads_f.permute(3, 0, 1, 2).unsqueeze(0)
        grads_g = grads_g.permute(3, 0, 1, 2).unsqueeze(0)

        cos_dist_n = F.cosine_similarity(dataset.n,
            grads_f, dim=1)
        cos_dist_b = F.cosine_similarity(dataset.b,
            grads_g, dim=1)

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        counts, bins = np.histogram(
            cos_dist_n.flatten().detach().cpu().numpy(), 
            bins=100,
            range=(-1.0, 1.0))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts,                 
            label=f"Avg error:  {(1-cos_dist_n.abs()).abs().mean().item() : 0.04f}")
        plt.title("Cosine similarity f gradient and N")
        plt.ylabel("Proportion")
        plt.xlabel("Cosine similarity")
        plt.legend()
        plt.show()

        counts, bins = np.histogram(
            cos_dist_b.flatten().detach().cpu().numpy(), 
            bins=100,
            range=(-1.0, 1.0))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts,
            label=f"Avg error:  {(1-cos_dist_b.abs()).abs().mean().item() : 0.04f}")
        plt.title("Cosine similarity g gradient and B")
        plt.ylabel("Proportion")
        plt.xlabel("Cosine similarity")
        plt.legend()
        plt.show()
        
        y_estimated = y_estimated.permute(3, 0, 1, 2).unsqueeze(0)
        create_folder(output_folder, opt['save_name'])
        tensor_to_cdf(y_estimated.detach(), 
            os.path.join(output_folder, opt['save_name'], 
            "dual_streamfunction.nc"))
     
    if(args['seeding_curve'] is not None):
        grid = list(dataset.data.shape[2:])
        with torch.no_grad():
            reconstructed_volume = model.sample_grid(grid)
        if(len(grid) == 3):
            reconstructed_volume = reconstructed_volume.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            reconstructed_volume = reconstructed_volume.permute(2, 0, 1).unsqueeze(0)
        
        f = reconstructed_volume[:,0:1]
        g = reconstructed_volume[:,1:2]
        
        f_output = f[0, 0, :, 80, 0]
        g_output = g[0, 0, :, 80, 0]
        
        B = torch.ones([f_output.shape[0], 1], device="cuda:0")
        
        higher_order = True
        
        if(higher_order):
            A = torch.ones([f_output.shape[0], 5], device="cuda:0")
            A[:,0] = f_output*f_output
            A[:,1] = f_output*g_output
            A[:,2] = g_output*g_output
            A[:,3] = f_output
            A[:,4] = g_output
        else:
            A = torch.ones([f_output.shape[0], 2], device="cuda:0")
            A[:,0] = f_output
            A[:,1] = g_output

        solution = torch.linalg.lstsq(A, B)
        
        if(higher_order):
            print(f"{solution.solution[0,0] : 0.04f}*(a^2) + " + \
                f"{solution.solution[1,0] : 0.04f}*(a*b) + " + \
                f"{solution.solution[2,0] : 0.04f}*(b^2) + " + \
                f"{solution.solution[3,0] : 0.04f}*a + " + \
                f"{solution.solution[4,0] : 0.04f}*b -1")
        else:
            print(f"{solution.solution[0,0] : 0.04f}*a + " + \
                f"{solution.solution[1,0] : 0.04f}*b -1")
            
        print("Residuals: ")
        print(solution.residuals)
        #t1 = torch.cat([n, b], dim=1)
        #tensor_to_cdf(t1, os.path.join(output_folder, "synth1_N_B.cdf"), ['n', 'b'])

    if(args['decompose'] is not None):
        scalar_potential = torch.zeros_like(dataset.data[2:]).unsqueeze(0)
        print(scalar_potential.shape)

    if(args['cai_method'] is not None):
        v = dataset.data
        b = binormal(v, normalize=False)
        n = normal(v, b=b, normalize=False)
        print(f"{b.min()}, {b.mean()}, {b.max()}")
        print(f"{n.min()}, {n.mean()}, {n.max()}")
        c = torch.zeros([1, 1, v.shape[2], v.shape[3], v.shape[4]],
            device=v.device, dtype=v.dtype)

        v_norm = v.norm(dim=1).unsqueeze(1)
        n_norm = n.norm(dim=1).unsqueeze(1)

        def PP(i, j, k):
            if(i == 0 and j == 0 and k == 0):
                return (0, 0, 0)
            elif(i == 0, j == 0):
                return (0, 0, k-1)
            elif(i == 0):
                return (0, j-1, k)
            else:
                return (i-1, j, k)
        deltap = 1/v.shape[2]
        for k in range(v.shape[2]):
            for j in range(v.shape[3]):
                for i in range(v.shape[4]):
                    print(f"{k},{j},{i}")
                    pp_i, pp_j, pp_k = PP(i, j, k)
                    v_spot = v[:,:,pp_k,pp_j,pp_i]
                    n_spot = n[:,:,pp_k,pp_j,pp_i]
                    fprimepp = v_spot.norm(dim=1)
                    fprimeprimepp = n_spot.norm(dim=1)
                    fp = c[:,:,pp_k, pp_j, pp_i] + \
                        (deltap * fprimepp) + \
                            (0.5*(deltap**2) * fprimeprimepp)
                    c[:,:,k,j,i] = fp
        tensor_to_cdf(c, "cai_test.nc")

    if(args['hausdorff_distance'] is not None):
        from vtk import vtkXMLPolyDataReader
        from vtkmodules.util.numpy_support import vtk_to_numpy
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(os.path.join(
            output_folder,
            args['load_from'],
            'original_streamlines.vtp'
        ))
        reader.Update()
        original_polyData = reader.GetOutput()
        original_points = original_polyData.GetPoints()
        original_cells = original_polyData.GetCellData()
        
        print(original_polyData.GetNumberOfCells())
        
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(os.path.join(
            output_folder,
            args['load_from'],
            'reconstructed_streamlines.vtp'
        ))
        reader.Update()
        reconstructed_polyData = reader.GetOutput()
        reconstructed_points = reconstructed_polyData.GetPoints()
        #array = points.GetData()        
        reconstructed_cells = reconstructed_polyData.GetCellData()
        
        print(reconstructed_polyData.GetNumberOfCells())
        
        IDs = vtk_to_numpy(original_cells.GetArray("SeedIds")).copy()
        IDs_reconstructed = vtk_to_numpy(reconstructed_cells.GetArray("SeedIds")).copy()
        print(IDs)
        print(IDs_reconstructed)
        from utility_functions import directed_hausdorff_nb
        
        
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[4,4])
        ax = fig.add_subplot(111, projection='3d')
        
        d_max = 0
        i_max = 0
        
        distances = []
        for i in range(original_polyData.GetNumberOfCells()):
            id = IDs[i]
            
            cell_original = original_polyData.GetCell(i)
            #print(cell_original)
            cell_reconstructed = reconstructed_polyData.GetCell(i)
            
            original_p = vtk_to_numpy(
                cell_original.GetPoints().GetData()).copy()
            reconstructed_p = vtk_to_numpy(
                cell_reconstructed.GetPoints().GetData()).copy()
            
            #print(original_p[0])
            #print(reconstructed_p[0])   
            #d = directed_hausdorff_nb(reconstructed_p, original_p)
            d1 = directed_hausdorff_nb(reconstructed_p, original_p)
            d2 = directed_hausdorff_nb(original_p, reconstructed_p)
            d = max(d1, d2)
            #print(f"{k}: original_p: {original_p.shape}, reconstructed_p: {reconstructed_p.shape}, dist {d : 0.05f}")
            distances.append(d)
            
            #ax.plot(reconstructed_p[:,0], reconstructed_p[:,1], reconstructed_p[:,2], color="blue")            
            #ax.plot(original_p[:,0], original_p[:,1], original_p[:,2], color="red")
            if(d > d_max):
                d_max = d
                i_max = i
        
        
        print(i_max)
        cell_original = original_polyData.GetCell(i_max)
        cell_reconstructed = reconstructed_polyData.GetCell(i_max)
        
        original_p = vtk_to_numpy(
            cell_original.GetPoints().GetData()).copy()
        reconstructed_p = vtk_to_numpy(
            cell_reconstructed.GetPoints().GetData()).copy()
        
        ax.plot(reconstructed_p[:,0], reconstructed_p[:,1], reconstructed_p[:,2], color="blue", label="reconstructed")            
        ax.plot(original_p[:,0], original_p[:,1], original_p[:,2], color="red", label="ground truth")
        ax.legend()
        plt.show()
         
         
        distances = np.array(distances)
        
        print(f"min/median/mean/average {distances.min() : 0.04f}/{np.median(distances) : 0.04f}/{distances.mean() : 0.04f}/{distances.max():0.04f}")
        
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        counts, bins = np.histogram(
            distances, 
            bins=100,
            range=(0.0, distances.max()))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts,
                label=f"min/median/mean/average {distances.min() : 0.02f}/{np.median(distances) : 0.02f}/{distances.mean() : 0.02f}/{distances.max():0.02f}")
        plt.title("Hausdorff Distances Btwn. Streamlines")
        plt.ylabel("Proportion")
        plt.xlabel("Hausdorff Distance")
        plt.legend()
        plt.show()
        
        plt.boxplot(distances, vert=False)
        plt.show()
        
        np.save("plume.numpy", distances)
        
    if(args['streamlines'] is not None):
        
        og = cdf_to_tensor(
            os.path.join(
                output_folder,
                args['load_from'],
                "original.nc"
            ),
            ['a','b','c']
        )[0].permute(1,2,3,0).numpy()
        
        reconstructed = cdf_to_tensor(
            os.path.join(
                output_folder,
                args['load_from'],
                "dual_streamfunction.nc"
            ),
            ['a','b','c']
        )[0].permute(1,2,3,0).numpy()
        
        if('vortices' in args['load_from']):
            seed_file = os.path.join(
                data_folder,
                "vortices_seeds.csv"
            )
        elif('cylinder' in args['load_from']):
            seed_file = os.path.join(
                data_folder,
                "cylinder_seeds.csv"
            )
        elif('ABC' in args['load_from']):
                seed_file = os.path.join(
                data_folder,
                "ABC_flow_seeds.csv"
            )
        elif('tornado' in args['load_from']):
                seed_file = os.path.join(
                data_folder,
                "tornado_seeds.csv"
            )
        elif('isabel' in args['load_from']):
                seed_file = os.path.join(
                data_folder,
                "isabel_seeds.csv"
            )
        elif('plume' in args['load_from']):
                seed_file = os.path.join(
                data_folder,
                "plume_seeds.csv"
            )
        
        
        seed_points = read_csv(seed_file, header=None).to_numpy()
        print(seed_points.shape)
        
        seeds = vtk.vtkPoints()
        seeds.SetData(numpy_support.numpy_to_vtk(seed_points.copy()))
        
        seeds_dataset = vtk.vtkPointSet()
        seeds_dataset.SetPoints(seeds)
        
        print(og.shape)
        print(reconstructed.shape)
        #og = np.transpose(og, (0, 2, 1, 3))
        #reconstructed = np.transpose(reconstructed, (0, 2, 1, 3))
        #og = np.flip(og, axis=-1)
        #reconstructed = np.flip(reconstructed, axis=-1)
        og_vtk_data = get_vtr(
            og.shape[:-1], 
            np.linspace(0, og.shape[2]-1, og.shape[2], dtype=np.float32), 
            np.linspace(0, og.shape[1]-1, og.shape[1], dtype=np.float32),
            np.linspace(0, og.shape[0]-1, og.shape[0], dtype=np.float32),
            vector_fields={'velocity': og.reshape(-1, 3)}
            )
        reconstructed_vtk_data = get_vtr(
            reconstructed.shape[:-1], 
            np.linspace(0, reconstructed.shape[2]-1, reconstructed.shape[2], dtype=np.float32), 
            np.linspace(0, reconstructed.shape[1]-1, reconstructed.shape[1], dtype=np.float32),
            np.linspace(0, reconstructed.shape[0]-1, reconstructed.shape[0], dtype=np.float32),
            vector_fields={'velocity': reconstructed.reshape(-1, 3)}
            )
        # set active vector for vtkStreamTracer
        og_vtk_data.GetPointData().SetActiveVectors('velocity')
        reconstructed_vtk_data.GetPointData().SetActiveVectors('velocity')
        
        # RungeKutta45 parameters
        init_steplen = 0.01
        tem_speed = 1e-12
        max_error = 1e-06
        min_intsteplen = 0.01
        max_intsteplen = 0.1
        max_steps = 10000
        max_length = 500+500+100

        gt_st = vtk.vtkStreamTracer()
        reconstructed_st = vtk.vtkStreamTracer()

        gt_st.SetInputData(og_vtk_data)
        gt_st.SetSourceData(seeds_dataset)
        reconstructed_st.SetInputData(reconstructed_vtk_data)
        reconstructed_st.SetSourceData(seeds_dataset)

        # integrator parameters
        gt_integrator = vtk.vtkRungeKutta45()
        gt_st.SetIntegrator(gt_integrator)
        gt_st.SetIntegrationDirectionToBoth()
        gt_st.SetMaximumError(max_error)
        gt_st.SetIntegrationStepUnit(gt_st.CELL_LENGTH_UNIT)
        gt_st.SetInitialIntegrationStep(init_steplen)
        gt_st.SetMinimumIntegrationStep(min_intsteplen)
        gt_st.SetMaximumIntegrationStep(max_intsteplen)
        gt_st.SetMaximumNumberOfSteps(max_steps)
        gt_st.SetMaximumPropagation(max_length)
        gt_st.SetTerminalSpeed(tem_speed)
        
        reconstructed_integrator = vtk.vtkRungeKutta45()
        reconstructed_st.SetIntegrator(reconstructed_integrator)
        reconstructed_st.SetIntegrationDirectionToBoth()
        reconstructed_st.SetMaximumError(max_error)
        reconstructed_st.SetIntegrationStepUnit(reconstructed_st.CELL_LENGTH_UNIT)
        reconstructed_st.SetInitialIntegrationStep(init_steplen)
        reconstructed_st.SetMinimumIntegrationStep(min_intsteplen)
        reconstructed_st.SetMaximumIntegrationStep(max_intsteplen)
        reconstructed_st.SetMaximumNumberOfSteps(max_steps)
        reconstructed_st.SetMaximumPropagation(max_length)
        reconstructed_st.SetTerminalSpeed(tem_speed)

        gt_st.Update()
        reconstructed_st.Update()

        og_streamlines = gt_st.GetOutput()
        reconstructed_streamlines = reconstructed_st.GetOutput()
        print(og_streamlines.GetNumberOfCells())
        print(reconstructed_streamlines.GetNumberOfCells())
        
        
        
        #IDs = vtk_to_numpy(original_cells.GetArray("SeedIds")).copy()
        #IDs_reconstructed = vtk_to_numpy(reconstructed_cells.GetArray("SeedIds")).copy()
        #print(IDs)
        #print(IDs_reconstructed)
        
        from utility_functions import directed_hausdorff_nb
        
        
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[7,7])
        ax = fig.add_subplot(111, projection='3d')
        
        d_max = 0
        i_max = 0
        
        distances = []
        for i in range(min(og_streamlines.GetNumberOfCells(), reconstructed_streamlines.GetNumberOfCells())):
            #id = IDs[i]
            
            cell_original = og_streamlines.GetCell(i)
            #print(cell_original)
            cell_reconstructed = reconstructed_streamlines.GetCell(i)
            
            original_p = vtk_to_numpy(
                cell_original.GetPoints().GetData()).copy()
            reconstructed_p = vtk_to_numpy(
                cell_reconstructed.GetPoints().GetData()).copy()
            
            #print(original_p[0])
            #print(reconstructed_p[0])   
            d1 = directed_hausdorff_nb(reconstructed_p, original_p)
            d2 = directed_hausdorff_nb(original_p, reconstructed_p)
            d = max(d1, d2)
            #d = directed_hausdorff_nb(original_p, reconstructed_p)
            print(f"{i}: original_p: {original_p.shape}, reconstructed_p: {reconstructed_p.shape}, dist {d : 0.05f}")
            distances.append(d)
            
            ax.plot(reconstructed_p[:,0], 
                    reconstructed_p[:,1], 
                    reconstructed_p[:,2], 
                    color="blue", alpha=0.2)            
            ax.plot(original_p[:,0], 
                    original_p[:,1], 
                    original_p[:,2], 
                    color="red", alpha=0.2)
            if(d > d_max):
                d_max = d
                i_max = i
        
        print(i_max)
        cell_original = og_streamlines.GetCell(i_max)
        cell_reconstructed = reconstructed_streamlines.GetCell(i_max)
        
        original_p = vtk_to_numpy(
            cell_original.GetPoints().GetData()).copy()
        reconstructed_p = vtk_to_numpy(
            cell_reconstructed.GetPoints().GetData()).copy()
        
        ax.plot(reconstructed_p[:,0], reconstructed_p[:,1], reconstructed_p[:,2], color="blue", label="reconstructed")            
        ax.plot(original_p[:,0], original_p[:,1], original_p[:,2], color="red", label="ground truth")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
         
         
        distances = np.array(distances)
        print(f"min/median/mean/average {distances.min() : 0.04f}/{np.median(distances) : 0.04f}/{distances.mean() : 0.04f}/{distances.max():0.04f}")
        
        
        import matplotlib.pyplot as plt
        
        plt.style.use('ggplot')
        counts, bins = np.histogram(
            distances, 
            bins=100,
            range=(0.0, distances.max()))
        counts = np.array(counts).astype(np.float32)
        counts /= counts.sum()
        plt.hist(bins[:-1], bins, weights=counts,
                label=f"min/median/mean/average {distances.min() : 0.02f}/{np.median(distances) : 0.02f}/{distances.mean() : 0.02f}/{distances.max():0.02f}")
        plt.title("Hausdorff Distances Btwn. Streamlines")
        plt.ylabel("Proportion")
        plt.xlabel("Hausdorff Distance")
        plt.legend()
        plt.show()
        
        plt.boxplot(distances, vert=False)
        plt.show()
        
        np.save("vortices.numpy", distances)
        
    if(args['boxplots'] is not None):
        abc = np.load(os.path.join(data_folder, "abc.npy")) / ((128**2 + 128**2 + 128**2)**0.5)
        vortices = np.load(os.path.join(data_folder, "vortices.npy"))/ ((128**2 + 128**2 + 128**2)**0.5)
        cylinder = np.load(os.path.join(data_folder, "cylinder.npy"))/ ((128**2 + 128**2 + 128**2)**0.5)
        tornado = np.load(os.path.join(data_folder, "tornado.npy"))/ ((128**2 + 128**2 + 128**2)**0.5)
        isabel = np.load(os.path.join(data_folder, "isabel.npy"))/ ((500**2 + 500**2 + 100**2)**0.5)
        plume = np.load(os.path.join(data_folder, "plume.npy"))    / ((1024**2 + 252**2 + 252**2)**0.5)
        
        d = [vortices, cylinder, abc, tornado, isabel, plume]
        
        from matplotlib import pyplot as plt
        
        plt.style.use('ggplot')
        font = {
            "font.size": 30
        }
        plt.rcParams.update(font)
        plt.boxplot(d, vert=False, labels=["Vortices", "Cylinder", "ABC", "Tornado", "Isabel", "Plume"])
        plt.title("Normalized Symmetric Hausdorff Distances")
        plt.xlabel("Voxel distance")
        plt.show()

    #writer.close()
        



        

