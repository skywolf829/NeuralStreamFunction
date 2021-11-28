from __future__ import absolute_import, division, print_function
import argparse
from datasets import Dataset
import datetime
from utility_functions import str2bool, PSNR, make_coord_grid, tensor_to_cdf, ssim3D
from models import load_model, save_model, ImplicitModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import h5py
from options import *
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

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
    parser.add_argument('--device',default="cuda:0",type=str)

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")


    if(args['load_from'] is None):
        print("Must load a model")
        quit()
         
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

    print(dataset.data.min())
    print(dataset.data.mean())
    print(dataset.data.max())

    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))

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
            tensor_to_cdf(supersampled_volume, os.path.join(output_folder, opt['save_name']+"_supersampled"))
            print(supersampled_volume.shape)
            p_model = PSNR(original_volume, supersampled_volume).item()
            s_model = ssim3D(original_volume, supersampled_volume).item()
            print("Neural network supersampling PSNR/SSIM: %0.03f/%0.05f" % (p_model, s_model))
            interpolated_volume = F.interpolate(dataset.data.to(opt['device']), size=original_volume.shape[2:],
                align_corners=False, mode='trilinear' if len(original_volume.shape) == 5 else 'bilinear')

            tensor_to_cdf(interpolated_volume, os.path.join(output_folder, opt['save_name']+"_interpolated"))
            p_interp = PSNR(original_volume, interpolated_volume).item()
            s_interp = ssim3D(original_volume, interpolated_volume).item()
            print("Interpolation supersampling PSNR: %0.03f/%0.05f" % (p_interp, s_interp))
            
            tensor_to_cdf(original_volume, os.path.join(output_folder, args['supersample_psnr']))

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

    writer.close()
        



        

