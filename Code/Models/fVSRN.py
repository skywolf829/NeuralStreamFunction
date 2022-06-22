import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Other.utility_functions import make_coord_grid
from Models.siren import SineLayer

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
            coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze().permute(1,0)
        fourier_features = self.pe(coords)
        feature_input = torch.cat([features, fourier_features], dim=1)
        output = self.net(feature_input)
        return output

    def forward_w_grad(self, coords):
        coords = coords.requires_grad_(True)
        output = self(coords)
        return output, coords
    
    def forward_maxpoints(self, coords, max_points=100000):
        print(coords.shape)
        output_shape = list(coords.shape)
        output_shape[-1] = self.opt['n_outputs']
        output = torch.empty(output_shape, 
            dtype=torch.float32, device=self.opt['device'])
        for start in range(0, coords.shape[0], max_points):
            #print("%i:%i" % (start, min(start+max_points, coords.shape[0])))
            output[start:min(start+max_points, coords.shape[0])] = \
                self(coords[start:min(start+max_points, coords.shape[0])])
        return output


    def sample_grid(self, grid, max_points = 100000):
        coord_grid = make_coord_grid(grid, self.opt['device'], False)
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
        vals = self.forward_maxpoints(coord_grid, max_points = 100000)
        coord_grid_shape[-1] = self.opt['n_outputs']
        vals = vals.reshape(coord_grid_shape)
        return vals

    def sample_grad_grid(self, grid, 
        output_dim = 0, max_points=1000):
        
        coord_grid = make_coord_grid(grid, 
            self.opt['device'], False)
        
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)       

        output_shape = list(coord_grid.shape)
        output_shape[-1] = self.opt['n_dims']
        print("Output shape")
        print(output_shape)
        output = torch.empty(output_shape, 
            dtype=torch.float32, device=self.opt['device'], 
            requires_grad=False)

        for start in range(0, coord_grid.shape[0], max_points):
            vals = self(
                coord_grid[start:min(start+max_points, coord_grid.shape[0])])
            grad = torch.autograd.grad(vals[:,output_dim], 
                coord_grid, 
                grad_outputs=torch.ones_like(vals[:,output_dim])
                )[0][start:min(start+max_points, coord_grid.shape[0])]
            
            output[start:min(start+max_points, coord_grid.shape[0])] = grad

        output = output.reshape(coord_grid_shape)
        
        return output

    def sample_grid_for_image(self, grid, boundary_scaling = 1.0):
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
            coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
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