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

        final_linear = nn.Linear(opt['nodes_per_layer'], 
                                 opt['n_outputs'], bias=True)
            
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / opt['nodes_per_layer']) / 30, 
                                            np.sqrt(6 / opt['nodes_per_layer']) / 30)
            
        self.net.append(final_linear)
        self.net.append(nn.BatchNorm1d(opt['n_outputs'], affine=False))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):     
        output = self.net(coords)
        return output

        