from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import os
from math import pi
from Models.options import *
from Other.utility_functions import create_folder
from Models.siren import SIREN
from Models.fSRN import fSRN
from Models.fVSRN import fVSRN
from Models.grid import Grid

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def save_model(model,opt):
    folder = create_folder(save_folder, opt["save_name"])
    path_to_save = os.path.join(save_folder, folder)
    
    torch.save({'state_dict': model.state_dict()}, 
        os.path.join(path_to_save, "model.ckpt.tar"),
        pickle_protocol=4
    )
    save_options(opt, path_to_save)

def load_model(opt, device):
    path_to_load = os.path.join(save_folder, opt["save_name"])
    model = create_model(opt)

    ckpt = torch.load(os.path.join(path_to_load, 'model.ckpt.tar'), 
        map_location = device)
    
    model.load_state_dict(ckpt['state_dict'])

    return model

def create_model(opt):
    if(opt['model'] == "siren"):
        return SIREN(opt)
    elif(opt['model'] == 'fSRN'):
        return fSRN(opt)
    elif(opt['model'] == 'fVSRN'):
        return fVSRN(opt)
    elif(opt['model'] == 'grid'):
        return Grid(opt)
    else:
        print(f"Model {opt['model']} does not exist.")
        quit()

class LReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(
                -torch.nn.init.calculate_gain("leaky_relu", 0.2),
                torch.nn.init.calculate_gain("leaky_relu", 0.2)
            )

    def forward(self, input):
        return F.leaky_relu(self.linear(input), 0.2)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.linear(input)
        return F.leaky_relu(intermediate, 0.2), intermediate

