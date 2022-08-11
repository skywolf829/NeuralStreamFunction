import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
other_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(other_dir)
from utility_functions import make_coord_grid
from siren import SineLayer

class PositionalEncoding(nn.Module):
    def __init__(self, opt):
        super(PositionalEncoding, self).__init__()        
        self.opt = opt
        self.L = opt['num_positional_encoding_terms']
        self.L_terms = torch.arange(0, opt['num_positional_encoding_terms'], 
            device=opt['device'], dtype=torch.float32).repeat_interleave(2*opt['n_dims'])
        self.L_terms = torch.pow(2, self.L_terms) * torch.pi

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
    
class SnakeAltLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, input):
        x = self.linear(input)
        return 0.5*x + torch.sin(x)**2

class fVSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.net = []        
        self.pe = PositionalEncoding(opt)
        self.feature_grid = torch.nn.Parameter(
            data = torch.empty([1, opt['n_features'], 
            opt['grid_size'],opt['grid_size'],opt['grid_size']], 
            device=self.opt['device'], dtype=torch.float32),
            requires_grad=True
        )

        self.net.append(SnakeAltLayer(opt['n_features'] + 
            opt['num_positional_encoding_terms']*opt['n_dims']*2, 
            opt['nodes_per_layer']))

        i = 0
        while i < opt['n_layers']:
            self.net.append(SnakeAltLayer(opt['nodes_per_layer'], opt['nodes_per_layer']))                 
            i += 1

        final_linear = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
            
        with torch.no_grad():
            nn.init.xavier_uniform_(final_linear.weight)
            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):   
        features = F.grid_sample(self.feature_grid, 
            coords.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            align_corners=True).squeeze().permute(1,0)
        fourier_features = self.pe(coords)
        feature_input = torch.cat([features, fourier_features], dim=1)
        output = self.net(feature_input)
        return output
        