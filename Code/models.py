from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import os
from math import pi
from options import *
from utility_functions import create_folder
import math
import numpy as np


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def save_model(model,opt):
    folder = create_folder(save_folder, opt["save_name"])
    path_to_save = os.path.join(save_folder, folder)
    
    torch.save(model.state_dict(), os.path.join(path_to_save, "model.ckpt"))
    save_options(opt, path_to_save)

def load_model(opt, device):
    path_to_load = os.path.join(save_folder, opt["save_name"], "model.ckpt")
    model = ImplicitModel(opt)
    params = torch.load(os.path.join(path_to_load, 'model.ckpt'), 
        map_location = device)
    model.load_state_dict(params)

    return model

class PositionalEncoding(nn.Module):
    def __init__(self, opt):
        super(PositionalEncoding, self).__init__()        
        self.opt = opt
        self.L = opt['num_positional_encoding_terms']
        self.L_terms = torch.arange(0, opt['num_positional_encoding_terms'], 
            device=opt['device'], dtype=torch.float32).repeat_interleave(2*opt['n_dims'])
        self.L_terms = torch.pow(2, self.L_terms) * pi

    def forward(self, locations):
        repeats = len(list(locations.shape)) * [1]
        repeats[-1] = self.L*2
        locations = locations.repeat(repeats)
        
        locations = locations * self.L_terms# + self.phase_shift
        if(self.opt['n_dims'] == 2):
            locations[..., 0::4] = torch.sin(locations[..., 0::4])
            locations[..., 1::4] = torch.sin(locations[..., 1::4])
            locations[..., 2::4] = torch.cos(locations[..., 2::4])
            locations[..., 3::4] = torch.cos(locations[..., 3::4])
        else:
            locations[..., 0::6] = torch.sin(locations[..., 0::6])
            locations[..., 1::6] = torch.sin(locations[..., 1::6])
            locations[..., 2::6] = torch.sin(locations[..., 2::6])
            locations[..., 3::6] = torch.cos(locations[..., 3::6])
            locations[..., 4::6] = torch.cos(locations[..., 4::6])
            locations[..., 5::6] = torch.cos(locations[..., 5::6])
        return locations

def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class ImplicitModel(nn.Module):
    def __init__ (self, opt):
        super(ImplicitModel, self).__init__()
        self.opt = opt
        self.activation_function = torch.sin if opt['activation_function'] \
            == 'sin' else torch.relu
        if(opt['use_positional_encoding']):
            self.positional_encoding = PositionalEncoding(opt)

        self.head = nn.Linear(opt['n_dims'] if not opt['use_positional_encoding'] else \
            opt['num_positional_encoding_terms'] * opt['n_dims'] * 2, opt['nodes_per_layer'])

        self.body = nn.ModuleList([])
        for _ in range(opt['n_layers']):
            self.body.append(nn.Linear(opt['nodes_per_layer'], opt['nodes_per_layer']))
        
        self.tail = nn.Linear(opt['nodes_per_layer'], opt['n_dims'])

        self.head.apply(first_layer_sine_init)
        self.body.apply(sine_init)
        self.tail.apply(sine_init)

    def forward(self,x):
        if(self.opt['use_positional_encoding']):
            x = self.positional_encoding(x)
            
        y_est = self.head(30 * x)
        y_est = self.activation_function(y_est)

        for i in range(len(self.body)):
            y_est = self.body[i](y_est)
            y_est = self.activation_function(30 * y_est)
        
        y_est = self.tail(y_est)
        return y_est
