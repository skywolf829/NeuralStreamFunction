import torch
import torch.nn as nn
import numpy as np
from Other.utility_functions import make_coord_grid
from Models.siren import SineLayer

class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
  

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)
    
class fSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.net = []

        self.net.append(SineLayer(opt['n_dims'], 
            opt['nodes_per_layer'], 
            is_first=True, omega_0=30))

        i = 0
        while i < opt['n_layers']:
            self.net.append(ResidualSineLayer(opt['nodes_per_layer'], 
                ave_first=i>0,
                ave_second=(i==opt['n_layers']-1),
                omega_0=30))                 
            i += 1

        final_linear = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
            
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / opt['nodes_per_layer']) / 30, 
                                            np.sqrt(6 / opt['nodes_per_layer']) / 30)
            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):     
        output = self.net(coords)
        return output

    def forward_w_grad(self, coords):
        coords = coords.requires_grad_(True)
        output = self.net(coords)
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
                self.net(coords[start:min(start+max_points, coords.shape[0])])
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
            vals = self.net(
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