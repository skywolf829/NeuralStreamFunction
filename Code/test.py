from __future__ import absolute_import, division, print_function
import argparse
from datasets import Dataset
import datetime
from utility_functions import str2bool, PSNR, make_coord_grid
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
        original_volume = h5py.File(os.path.join(data_folder, args['original_volume']), 'r')
        original_volume = torch.tensor(original_volume).to(opt['device']).unsqueeze(0)
        grid = list(original_volume.shape[2:])
        with torch.no_grad():
            supersampled_volume = model.sample_grid(grid)
            p_model = PSNR(original_volume, supersampled_volume).item()
            print("Neural network supersampling PSNR: %0.03f" % p_model)
            interpolated_volume = F.interpolate(dataset.data.to(opt['device']), size=original_volume.shape[2:],
            align_corners=False, mode='trilinear' if len(original_volume.shape) == 5 else 'bilinear')
            p_interp = PSNR(original_volume, supersampled_volume).item()
            print("Interpolation supersampling PSNR: %0.03f" % p_interp)
            

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
        



        

