from __future__ import absolute_import, division, print_function
import argparse
import os
import sys
script_dir = os.path.dirname(__file__)
other_dir = os.path.join(script_dir, "..", "Other")
models_dir = os.path.join(script_dir, "..", "Models")
datasets_dir = os.path.join(script_dir, "..", "Datasets")
sys.path.append(other_dir)
sys.path.append(models_dir)
sys.path.append(datasets_dir)
sys.path.append(script_dir)
from utility_functions import PSNR, tensor_to_cdf, create_path, particle_tracing, visualize_traces, make_coord_grid, curl
from models import load_model, sample_grid, sample_grad_grid, forward_maxpoints
from options import load_options
import torch.nn.functional as F
from datasets import Dataset
import torch
import time
import numpy as np

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def model_reconstruction(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    if("dsfm" in opt['training_mode']): 
        grads_f = sample_grad_grid(model, grid, output_dim=0, 
                                        max_points=10000)
        grads_g = sample_grad_grid(model, grid, output_dim=1, 
                                        max_points=10000)
        grads_f = grads_f.permute(3, 0, 1, 2).unsqueeze(0)
        grads_g = grads_g.permute(3, 0, 1, 2).unsqueeze(0)
        with torch.no_grad():
            m = sample_grid(model, grid, max_points=10000)[...,2:3]
            m = m.permute(3, 0, 1, 2).unsqueeze(0)
        result = torch.cross(grads_f, grads_g, dim=1)
        result /= (result.norm(dim=1) + 1e-8)
        result *= m
        
    elif("uvw" in opt['training_mode']):
        with torch.no_grad():
            result = sample_grid(model, grid, max_points = 10000)
            result = result[...,0:3]
            result = result.permute(3, 0, 1, 2).unsqueeze(0)
            
    result = result.to(opt['data_device'])

    p = PSNR(result, data.data)

    print(f"PSNR: {p : 0.02f}")
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(result, os.path.join(output_folder, "Reconstruction", opt['save_name']+".nc"))

def model_stream_function(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    total_points = 1
    for i in range(len(grid)):
        total_points *= grid[i]
    
    if("dsf" in opt['training_mode'] or opt['training_mode'].startswith("f_") \
        or opt['training_mode'] == "vorticity" or opt['training_mode'] == "APAP"):          
        t_0_ff = time.time()
        with torch.no_grad():
            f = sample_grid(model, grid, max_points=10000)[...,0:1]
            t_1_ff = time.time()
            f = f.permute(3, 0, 1, 2).unsqueeze(0)   
            GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
            / (1024**3))
        
        t_0_ff_w_grad = time.time()
        f_grad = sample_grad_grid(model, grid, output_dim=0, max_points=10000)
        t_1_ff_w_grad = time.time()
        f_grad = f_grad.permute(3,0,1,2).unsqueeze(0)
        GBytes_w_grad = (torch.cuda.max_memory_allocated(device=opt['device']) \
            / (1024**3))
        
    elif("uvwf" in opt['training_mode']):
        with torch.no_grad():
            f = sample_grid(model, grid, max_points = 2097152)[...,3:4]
            f = f.permute(3, 0, 1, 2).unsqueeze(0)
        f_grad = sample_grad_grid(model, grid, output_dim=3, max_points=2097152)
        f_grad = f_grad.permute(3,0,1,2).unsqueeze(0)

    elif("PSF" in opt['training_mode']):
        with torch.no_grad():
            f = sample_grid(model, grid, max_points = 2097152).permute(3,0,1,2).unsqueeze(0)
            
            GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
            / (1024**3))
            coord_grid = make_coord_grid(grid, 
                model.opt['device'], flatten=False,
                align_corners=model.opt['align_corners'])
            coord_grid_shape = list(coord_grid.shape)
            coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
            vals = model.forward_grad(coord_grid)
            coord_grid_shape[-1] = model.opt['n_outputs']*3
            vals = vals.reshape(coord_grid_shape)
            f_grad = vals.permute(3,0,1,2).unsqueeze(0)
            GBytes_w_grad = (torch.cuda.max_memory_allocated(device=opt['device']) \
            / (1024**3))
    
    f = f.to(opt['data_device'])
    f_grad = f_grad.to(opt['data_device'])
    d = dataset.data
    
    cos_dist = F.cosine_similarity(d,
            f_grad, dim=1)
    cos_dist = torch.clamp(cos_dist, min=-1 + 1E-6, max=1-1E-6)
    angles = torch.acos(cos_dist)*(180/torch.pi)
    angles = torch.abs(90-angles)
    
    #import matplotlib.pyplot as plt
    #plt.boxplot(angles.cpu().numpy().flatten(), vert=False, showfliers=False)
    #plt.show()
    create_path(os.path.join(output_folder, "Error"))
    np.save(os.path.join(output_folder, "Error", opt['save_name']+".npy"),
        angles.cpu().numpy().flatten())
    print(f"Minimum angle error off perpendicular {angles.min().item() : 0.03f} deg.")
    print(f"Median angle error off perpendicular {angles.median().item() : 0.03f} deg.")
    print(f"Average angle error off perpendicular {angles.mean().item() : 0.03f} deg.")
    print(f"Maximum angle error off perpendicular {angles.max().item() : 0.03f} deg.")
    GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
            / (1024**3))
    #print(f"Inference took {t_1_ff-t_0_ff: 0.04f} sec. for grid {grid} with {total_points} points. {(t_1_ff-t_0_ff)/total_points: 0.09f} sec. per point")
    #print(f"Inference with grad took {t_1_ff_w_grad-t_0_ff_w_grad: 0.04f} sec. for grid {grid} with {total_points} points. {(t_1_ff_w_grad-t_0_ff_w_grad)/total_points: 0.09f} sec. per point")

    print(f"Maximum memory allocated on {opt['device']} was {GBytes : 0.02f} GB")
    print(f"Maximum memory allocated w/ grad on {opt['device']} was {GBytes_w_grad : 0.02f} GB")
    tensor_to_cdf(angles.unsqueeze(0), 
                  os.path.join(output_folder, "StreamFunction", opt['save_name']+"_error.nc"))
    
    create_path(os.path.join(output_folder, "StreamFunction"))
    
    print(f"Min stream function value {f.min().item() : 0.03f}.")
    print(f"Max stream function value {f.max().item() : 0.03f}.")
    tensor_to_cdf(f, os.path.join(output_folder, "StreamFunction", opt['save_name']+".nc"))

def model_streamline_error(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    total_points = 1
    for i in range(len(grid)):
        total_points *= grid[i]
    
    traces = particle_tracing(dataset.data,
        dataset.seeds, steps=500, h=0.75, 
        align_corners=opt['align_corners'])
    
    #visualize_traces(traces)
    
    trace_shape = list(traces.shape)
    trace_shape[-1] = 1
    model_trace_output = model(traces.reshape(-1, 3))
    model_trace_output = model_trace_output.reshape(trace_shape)
    model_trace_output = torch.abs(model_trace_output)
    
    print(f"Average streamfunction error for traces: {model_trace_output.mean()}")
    print(f"Max streamfunction error for traces: {model_trace_output.max()}")

def perform_tests(model, data, tests, opt):
    if("reconstruction" in tests):
        if("dsfm" in opt['training_mode'] or
            "uvw" in opt['training_mode'] or
            opt['training_mode'] == "hhd"):
            model_reconstruction(model, data, opt)
        else:
            print(f"Training mode {opt['training_mode']} does not support the reconstruction task")
    if("streamfunction" in tests):
        if("dsf" in opt['training_mode'] or \
            "uvwf" in opt['training_mode'] or
            "f_" in opt['training_mode'] or "PSF" in opt['training_mode'] \
                or "vorticity" in opt['training_mode'] or "APAP" in opt['training_mode']):
            model_stream_function(model, data, opt)
        else:
            print(f"Training mode {opt['training_mode']} does not support the stream function task")
    if("streamline" in tests):
        model_streamline_error(model, data, opt)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--tests_to_run',default=None,type=str,
                        help="A set of tests to run, separated by commas - in our final paper, only streamfunction is relevant")
    parser.add_argument('--device',default=None,type=str,
                        help="Device to load model to")
    parser.add_argument('--data_device',default=None,type=str,
                        help="Device to load data to")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    tests_to_run = args['tests_to_run'].split(',')
    
    # Load the model
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    opt['data_device'] = args['data_device']
    model = load_model(opt, args['device']).to(args['device'])
    model.eval()
    
    # Load the reference data
    data = Dataset(opt)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters {n_params}")
    
    # Perform tests
    perform_tests(model, data, tests_to_run, opt)
    
        
    
        



        

