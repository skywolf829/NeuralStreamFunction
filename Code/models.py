from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import os
from math import pi
from options import *
from utility_functions import create_folder, make_coord_grid
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
    path_to_load = os.path.join(save_folder, opt["save_name"])
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
    

class SineLayer(nn.Module):  
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class ImplicitModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.net = []
        if(self.opt['periodic']):
            self.factor =  torch.tensor(pi/2, dtype=torch.float32, 
                device = opt['device'])

        self.net.append(SineLayer(opt['n_dims'], opt['nodes_per_layer'], 
                                  is_first=True, omega_0=30))

        for i in range(opt['n_layers']):
            self.net.append(SineLayer(opt['nodes_per_layer'], opt['nodes_per_layer'], 
                                      is_first=False, omega_0=30))

        final_linear = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
            
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / opt['nodes_per_layer']) / 30, 
                                            np.sqrt(6 / opt['nodes_per_layer']) / 30)
            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):        
        if(self.opt['periodic']):
            #coords = torch.sin(coords * self.factor)
            coords = -self.factor * torch.arctan(1/(torch.tan(self.factor * (coords-1))))
        output = self.net(coords)
        return output
    
    def forward_maxpoints(self, coords, max_points=10000):
        print(coords.shape)
        if(self.opt['periodic']):
            #coords = torch.sin(coords * self.factor)
            coords = -self.factor * torch.arctan(1/(torch.tan(self.factor * (coords-1))))
            
        output_shape = list(coords.shape)
        output_shape[-1] = self.opt['n_outputs']
        output = torch.empty(output_shape, 
            dtype=torch.float32, device=self.opt['device'])
        for start in range(0, coords.shape[0], max_points):
            #print("%i:%i" % (start, min(start+max_points, coords.shape[0])))
            output[start:min(start+max_points, coords.shape[0])] = \
                self.net(coords[start:min(start+max_points, coords.shape[0])])
        return output


    def sample_grid(self, grid):
        coord_grid = make_coord_grid(grid, self.opt['device'], False)
        
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
        vals = self.forward_maxpoints(coord_grid)
        coord_grid_shape[-1] = self.opt['n_outputs']
        vals = vals.reshape(coord_grid_shape)
        return vals

    def sample_grad_grid(self, grid=None, coord_grid=None, input_dim = None, output_dim = None):
        if(coord_grid is None and grid is not None):
            coord_grid = make_coord_grid(grid, self.opt['device'], False)
        elif(coord_grid is None and grid is None):
            coord_grid = make_coord_grid(self.dataset.data.shape[2:])
        
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)
        vals = self.forward_maxpoints(coord_grid)        

        grad = list(
            torch.autograd.grad(vals, coord_grid, grad_outputs=torch.ones_like(vals)))

        coord_grid_shape[-1] = self.opt['n_dims']
        for i in range(len(grad)):
            grad[i] = grad[i].reshape(coord_grid_shape)
        
        return grad

    def sample_grid_for_image(self, grid, boundary_scaling = 1.0):
        coord_grid = make_coord_grid(grid, self.opt['device'], False)
        print(coord_grid.shape)
        if(len(coord_grid.shape) == 4):
            coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
        
        coord_grid *= boundary_scaling

        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
        vals = self.forward_maxpoints(coord_grid)
        coord_grid_shape[-1] = self.opt['n_outputs']
        vals = vals.reshape(coord_grid_shape)
        if(self.opt['loss'] == "l1occupancy"):
            vals = vals[..., 0:-1]
        return vals

    def sample_occupancy_grid_for_image(self, grid, boundary_scaling = 1.0):
        coord_grid = make_coord_grid(grid, self.opt['device'], False)
        if(len(coord_grid.shape) == 4):
            coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
        
        coord_grid *= boundary_scaling

        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
        vals = self.forward_maxpoints(coord_grid)
        coord_grid_shape[-1] = self.opt['n_outputs']
        vals = vals.reshape(coord_grid_shape)
        if(self.opt['loss'] == "l1occupancy"):
            vals = vals[...,-1]
        return vals
    
    def sample_grad_grid_for_image(self, grid, boundary_scaling = 1.0, 
        input_dim = 0, output_dim = 0):

        coord_grid = make_coord_grid(grid, self.opt['device'], False)        
        if(len(coord_grid.shape) == 4):
            coord_grid = coord_grid[:,:,:,:int(coord_grid.shape[3]/2)]
        coord_grid *= boundary_scaling
        
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)
        vals = self.forward_maxpoints(coord_grid)    


        grad = torch.autograd.grad(vals[:,output_dim], 
            coord_grid,#[:,input_dim], 
            grad_outputs=torch.ones_like(vals[:, output_dim]),
            allow_unused=True)
        

        grad = grad[0][:,input_dim]
        coord_grid_shape[-1] = 1
        grad = grad.reshape(coord_grid_shape)
        
        return grad
    
    def sample_rect(self, starts, widths, samples):
        positions = []
        for i in range(len(starts)):
            positions.append(
                torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                    dtype=torch.float32, device=self.opt['device'])
            )
        grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1)
        vals = self.forward(grid_to_sample)
        return vals

    def sample_grad_rect(self, starts, widths, samples, input_dim, output_dim):
        positions = []
        for i in range(len(starts)):
            positions.append(
                torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                    dtype=torch.float32, device=self.opt['device'])
            )
        grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1).requires_grad_(True)
        vals = self.forward(grid_to_sample)
        
        grad = torch.autograd.grad(vals[:,output_dim], 
            grid_to_sample,#[:,input_dim], 
            grad_outputs=torch.ones_like(vals[:, output_dim]),
            allow_unused=True)
        

        grad = grad[0][:,input_dim]
        grid_to_sample[-1] = 1
        grad = grad.reshape(grid_to_sample)

        return vals