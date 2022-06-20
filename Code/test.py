from __future__ import absolute_import, division, print_function
import argparse
import os
from Other.utility_functions import PSNR, tensor_to_cdf, create_path
from Models.models import load_model
from Models.options import *
from Datasets.datasets import Dataset
import torch

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def model_reconstruction(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    if("dsfm" in opt['training_mode']): 
        grads_f = model.sample_grad_grid(grid, output_dim=0, 
                                        max_points=10000)
        grads_g = model.sample_grad_grid(grid, output_dim=1, 
                                        max_points=10000)
        grads_f = grads_f.permute(3, 0, 1, 2).unsqueeze(0)
        grads_g = grads_g.permute(3, 0, 1, 2).unsqueeze(0)
        with torch.no_grad():
            m = model.sample_grid(grid, max_points=100000)[...,2:3]
            m = m.permute(3, 0, 1, 2).unsqueeze(0)
        print(grads_f.shape)
        print(grads_g.shape)
        print(m.shape)
        result = torch.cross(grads_f, grads_g, dim=1)
        print(result.shape)
        result /= (result.norm(dim=1) + 1e-8)
        print(result.shape)
        result *= m
        print(result.shape)
        
    elif("uvw" in opt['training_mode']):
        with torch.no_grad():
            result = model.sample_grid(grid, max_points = 100000)
            print(result.shape)
            result = result[...,0:3]
            print(result.shape)
            result = result.permute(3, 0, 1, 2).unsqueeze(0)
            print(result.shape)
            
    result = result.to(opt['data_device'])
    p = PSNR(result, data.data)
    print(f"PSNR: {p : 0.02f}")
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(result, os.path.join(output_folder, "Reconstruction", opt['save_name']+".nc"))

def model_stream_function(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    if("dsfm" in opt['training_mode']): 
        with torch.no_grad():
            f = model.sample_grid(grid, max_points=100000)[...,0:1]
            f = f.permute(3, 0, 1, 2).unsqueeze(0)
        
    elif("uvwf" in opt['training_mode']):
        with torch.no_grad():
            f = model.sample_grid(grid, max_points = 100000)[...,3:4]
            f = f.permute(3, 0, 1, 2).unsqueeze(0)
    
    f = f.to(opt['data_device'])
    create_path(os.path.join(output_folder, "StreamFunction"))
    tensor_to_cdf(f, os.path.join(output_folder, "StreamFunction", opt['save_name']+".nc"))

def perform_tests(model, data, tests, opt):
    if("reconstruction" in tests):
        if(opt['training_mode'] == "dsfm_any" or
            opt['training_mode'] == "dsfm_parallel" or
            opt['training_mode'] == "dsfm_direction" or
            opt['training_mode'] == "uvw" or 
            opt['training_mode'] == "uvwf_any" or
            opt['training_mode'] == "uvwf_parallel" or
            opt['training_mode'] == "uvwf_direction" or
            opt['training_mode'] == "hhd"):
            model_reconstruction(model, data, opt)
        else:
            print(f"Training mode {opt['training_mode']} does not support the reconstruction task")
    if("streamfunction" in tests):
        if("dsf" in opt['training_mode'] or \
            "uvwf" in opt['training_mode']):
            model_stream_function(model, data, opt)
        else:
            print(f"Training mode {opt['training_mode']} does not support the stream function task")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--tests_to_run',default=None,type=str,
                        help="A set of tests to run, separated by commas")
    parser.add_argument('--device',default=None,type=str,
                        help="Device to load model to")
    parser.add_argument('--data_device',default=None,type=str,
                        help="Device to load data to")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
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
    
    # Perform tests
    perform_tests(model, data, tests_to_run, opt)
    
        
    
        



        

