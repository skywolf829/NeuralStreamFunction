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
from utility_functions import jacobian, curl, nc_to_tensor, tensor_to_cdf, create_path
import numpy as np
import torch.nn.functional as F
import torch

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--data',default=None,type=str,help="Model name to load")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    vector_field = nc_to_tensor(os.path.join(data_folder, args['data']))
    PSS_result = nc_to_tensor(os.path.join((output_folder), 
        "PSS",
        args['data']))[:,0:1]
    print(vector_field.shape)
    print(PSS_result.shape)
    #inside_fluids_result = F.interpolate(PSS_result,
    #    size = vector_field.shape[2:], mode="trilinear", align_corners=False)

    #vorticity_field = curl(vector_field)
    sx_grad = jacobian(PSS_result, False)[0]

    cos_dist = F.cosine_similarity(vector_field,
            sx_grad, dim=1)
    cos_dist = torch.clamp(cos_dist, min=-1 + 1E-6, max=1-1E-6)
    angles = torch.acos(cos_dist)*(180/torch.pi)
    print(f"Maximum angles dist {angles.max().item() : 0.03f} deg.")
    print(f"Minimum angles dist {angles.min().item() : 0.03f} deg.")
    angles = torch.abs(90-angles)
    create_path(os.path.join(output_folder, "Error"))
    np.save(os.path.join(output_folder, "Error", args["data"]+"_PSS.npy"),
        angles.cpu().numpy().flatten())

    print(f"Minimum angle error off perpendicular {angles.min().item() : 0.03f} deg.")
    print(f"Maximum angle error off perpendicular {angles.max().item() : 0.03f} deg.")
    print(f"Average angle error off perpendicular {angles.mean().item() : 0.03f} deg.")
    print(f"Median angle error off perpendicular {angles.median().item() : 0.03f} deg.")
    
    tensor_to_cdf(angles.unsqueeze(0), 
                  os.path.join(output_folder, "PSS", args['data']+"_error.nc"),
                  "error")
    
    
        
    
        



        

