import os
import sys
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "..", "Models")
sys.path.append(models_dir)
utility_fn_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(utility_fn_dir)
datasets_dir = os.path.join(script_dir, "..", "Datasets")
sys.path.append(datasets_dir)
import torch
import argparse
from options import load_options
from models import load_model, forward_w_grad, forward_maxpoints, sample_grid
from datasets import get_dataset
from utility_functions import nc_to_tensor
import time
from typing import List
import imageio
import numpy as np
import torch.nn

def tensor_to_img(t: torch.Tensor, path:str):    
    img_data: np.ndarray = t.detach().cpu().numpy()
    img_data *= 255
    img_data = img_data.astype(np.uint8)
    imageio.imwrite(path, img_data)

class Grid():
    """
    A class used to represent a grid
    """

    def __init__(self, 
        z_min: float, z_max: float, z_dim: int,
        y_min: float, y_max: float, y_dim: int,
        x_min: float, x_max: float, x_dim: int):
        """
        the initialization method for a
        Grid object. Requires the extents and dimension sizes.
        """
        self.__zmin : float = z_min
        self.__zmax : float = z_max
        self.__zdim : float = z_dim
        self.__ymin : float = y_min
        self.__ymax : float = y_max
        self.__ydim : float = y_dim
        self.__xmin : float = x_min
        self.__xmax : float = x_max
        self.__xdim : float = x_dim
        
    def get_extents(self) -> List[float]:
        '''
        returns the extents for the instantiated Grid
        in the same order that the initialization method takes.
        '''
        return [
            self.__zmin, self.__zmax,
            self.__ymin, self.__ymax,
            self.__xmin, self.__xmax
        ]

    def shape(self) -> List[int]:
        '''
        Returns the z,y,x dim sizes in that order
        '''
        return [
            self.__zdim,
            self.__ydim,
            self.__xdim
        ]

    def in_bounds(self, positions: torch.Tensor):
        '''
        checks whether positions (given in z, y, x order)
        is within the domain of the defined grid or
        not. positions is a tensor of size [n, 3], equating 
        to n points to check if they are in
        the bounds.
        '''
        in_bounds : torch.Tensor = positions[:,0] >= self.__zmin
        in_bounds : torch.Tensor = torch.logical_and(in_bounds, 
            (positions[:,0] <= self.__zmax))
            
        in_bounds : torch.Tensor = torch.logical_and(in_bounds, 
            positions[:,1] >= self.__ymin)
        in_bounds : torch.Tensor = torch.logical_and(in_bounds, 
            positions[:,1] <= self.__ymax)

        in_bounds : torch.Tensor = torch.logical_and(in_bounds, 
            positions[:,2] >= self.__xmin)
        in_bounds : torch.Tensor = torch.logical_and(in_bounds, 
            positions[:,2] <= self.__xmax)
        return in_bounds#.all()
    
    def index_and_weights_for(self, positions): 
        """
        takes an array of positions of shape [n, 3], and returns their
        z,y,x indices (in return position 0) and trilinear interpolation 
        weights for the 8 corners (in return position 1). 
        Return position 0 is of shape [n, 3] (3 indices per position) 
        and return position 1 is of shape [n, 8] (8 trilinear 
        weights per position). 
        """
        indices : torch.Tensor = positions - torch.tensor(
            [self.__zmin, self.__ymin, self.__xmin], 
            device = positions.device
            ).unsqueeze(0).repeat(positions.shape[0], 1)
        indices /= torch.tensor(
            [self.__zmax - self.__zmin, 
            self.__ymax - self.__ymin, 
            self.__xmax - self.__xmin], 
            device = positions.device
            ).unsqueeze(0).repeat(positions.shape[0], 1)
        indices *= torch.tensor(
            [self.__zdim-1, self.__ydim-1, self.__xdim-1],
            device = positions.device
            ).unsqueeze(0).repeat(positions.shape[0], 1)

        indices_floor : torch.Tensor = indices.clone().type(dtype=torch.long)

        diffs : torch.Tensor = indices - indices_floor
        weights : torch.Tensor = torch.empty(size=[positions.shape[0], 8],
                            device=positions.device)
        weights[:,0] = (1-diffs[:,0])*(1-diffs[:,1])*(1-diffs[:,2])
        weights[:,1] = diffs[:,0]*(1-diffs[:,1])*(1-diffs[:,2])
        weights[:,2] = (1-diffs[:,0])*diffs[:,1]*(1-diffs[:,2])
        weights[:,3] = diffs[:,0]*diffs[:,1]*(1-diffs[:,2])
        weights[:,4] = (1-diffs[:,0])*(1-diffs[:,1])*diffs[:,2]
        weights[:,5] = diffs[:,0]*(1-diffs[:,1])*diffs[:,2]
        weights[:,6] = (1-diffs[:,0])*diffs[:,1]*diffs[:,2]
        weights[:,7] = diffs[:,0]*diffs[:,1]*diffs[:,2]
        
        return indices_floor, weights

class Solution():
    '''
    A solution class that contains the scalar/multivariate field data 
    represented as a 3D array.
    '''

    def __init__(self, 
        chans=None,
        zdim=None, 
        ydim=None, xdim=None,
        data=None, device=None):
        self.__data : torch.Tensor = torch.tensor([0])
        self.__device : str = ""
        self.__channels : int = 0
        self.__zdim : int = 0
        self.__ydim : int = 0
        self.__xdim : int = 0
        self.__data_grad : torch.Tensor = torch.tensor([0])
        self.__did_precompute_gradients = False
        if device is not None:
            self.__device = device
        else:
            self.__device = "cpu"
            
        if data is not None:
            self.__data  = data.clone()
            self.__device= str(data.device)
            self.__zdim = int(data.shape[0])
            self.__ydim  = int(data.shape[1])
            self.__xdim  = int(data.shape[2])
            self.__channels = int(data.shape[3])
            #print("Precomputing all gradients")
            #self.__data_grad = self.precompute_gradients()
            #self.__did_precompute_gradients = True
        else:
            if(chans is not None):
                self.__channels = chans
            else:
                self.__channels = 1
            if(zdim is not None):
                self.__zdim= zdim
            else:
                self.__zdim = 2
            if(ydim is not None):
                self.__ydim= ydim
            else:
                self.__ydim = 2
            if(xdim is not None):
                self.__xdim = xdim
            else:
                self.__xdim = 2
            self.__data = torch.zeros((
                self.__zdim, self.__ydim, self.__xdim,
                self.__channels), 
                dtype=torch.float32,
                device=self.__device)

    def device(self) -> str:
        return self.__device

    def to_device(self, device: str):
        self.__device : str = device
        self.__data : torch.Tensor = self.__data.to(device)
        self.__data_grad : torch.Tensor = self.__data_grad.to(device)

    def set_dims(self, z_size: int, y_size: int, x_size: int):
        '''
        Assigns the z,y,x dimension lengths
        '''
        self.__zdim: int = z_size
        self.__ydim: int = y_size
        self.__xdim: int = x_size
        self.__data: torch.Tensor = torch.empty(
            [self.__zdim, self.__ydim, self.__xdim,
             self.__channels], 
            device=self.__device)

    def shape(self):
        '''
        Returns the # variables and dimension lengths in z,y,x order
        '''
        return (self.__zdim, self.__ydim, self.__xdim, self.__channels)

    def get_min(self) -> float:
        return self.__data.min().item()

    def get_max(self) -> float:
        return self.__data.max().item()

    def set_data(self, data: torch.Tensor):      
        self.__zdim : int = data.shape[0]
        self.__ydim : int = data.shape[1]
        self.__xdim : int = data.shape[2]
        self.__channels : int = data.shape[3]  
        self.__data : torch.Tensor = data.clone()
        self.__device : str = str(data.device)

    def get_data(self) -> torch.Tensor:
        return self.__data

    def solution_at(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Returns the solution at grid points
        '''
        assert positions.dtype is torch.long
                
        #print(positions.chunk(3,1)[0].shape)
        #print(self.__data.shape)
        solutions : torch.Tensor = self.__data[positions[:,0],
                                               positions[:,1],
                                               positions[:,2],
                                               :]
        #solutions : torch.Tensor = self.__data[positions.chunk(3, 1)]
        return solutions

    def set_at(self, positions: torch.Tensor, v : torch.Tensor):
        '''
        Sets the solution at a grid point
        '''
        assert positions.dtype is torch.long
        #self.__data[positions.chunk(chunks=3, dim=1)] = v    
        self.__data[positions[:,0],
                                positions[:,1],
                                positions[:,2],
                                :] = v
    
    def precompute_gradients(self):
        kji = torch.meshgrid(
            [
                torch.tensor(np.linspace(0, self.__data.shape[0]-1, num=self.__data.shape[0])),
                torch.tensor(np.linspace(0, self.__data.shape[1]-1, num=self.__data.shape[1])),
                torch.tensor(np.linspace(0, self.__data.shape[2]-1, num=self.__data.shape[2]))
            ],
            indexing='ij'
        )
        kji = torch.stack(kji)        
        kji = kji.flatten(1).permute(1,0).type(torch.long)
        
        grads = self.grad_at(kji)
        grads = grads.reshape(self.__data.shape[1],
                              self.__data.shape[2],
                              self.__data.shape[3], 
                              3*self.__channels)
        return grads
        
    def grad_at(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Computes the gradient at a grid point using first order central 
        difference when in the interior of the volume, and using forward
        or backward difference when on the boundary.
        '''
        assert self.__zdim > 1 and self.__ydim > 1 and self.__xdim > 1
        assert positions.dtype is torch.long
        grads : torch.Tensor = torch.empty(
            [positions.shape[0], 3*self.__channels], 
            device=self.__device)

        if(self.__did_precompute_gradients):
            #grads = self.__data_grad[positions.chunk(3,1)]
            grads = self.__data_grad[positions[:,0],
                                               positions[:,1],
                                               positions[:,2],
                                               :]
            #grads = grads[:,0,:]
        else:
            # z derivative
            forward_diff_indices : torch.Tensor = positions[:,0] == 0
            backward_diff_indices : torch.Tensor = positions[:,0] == self.__zdim-1
            central_diff_indices : torch.Tensor = torch.logical_and(
                positions[:,0] > 0, positions[:,0] < self.__zdim-1)
            
            grads[forward_diff_indices, 0:self.__channels] = \
                (self.solution_at(
                    positions[forward_diff_indices]
                            + torch.tensor([1,0,0], device=self.__device)
                ) - self.solution_at(
                    positions[forward_diff_indices]
                ))*(1/self.__zdim)                      
            grads[backward_diff_indices, 0:self.__channels] = \
                (-self.solution_at(
                    positions[backward_diff_indices]
                            - torch.tensor([1,0,0], device=self.__device)
                ) + self.solution_at(
                    positions[backward_diff_indices]
                ))*(1/self.__zdim)  
                
            grads[central_diff_indices, 0:self.__channels] = \
                (self.solution_at(
                    positions[central_diff_indices]
                            + torch.tensor([1,0,0], device=self.__device)
                ) - (self.solution_at(
                    positions[central_diff_indices]
                    - torch.tensor([1,0,0], device=self.__device))
                ))*(1/self.__zdim)  

            # y derivative
            forward_diff_indices : torch.Tensor = positions[:,1] == 0
            backward_diff_indices : torch.Tensor = positions[:,1] == self.__ydim-1
            central_diff_indices : torch.Tensor = torch.logical_and(
                positions[:,1] > 0,
                positions[:,1] < self.__ydim-1)
            
            grads[forward_diff_indices, self.__channels:2*self.__channels] = \
                (self.solution_at(
                    positions[forward_diff_indices]
                            + torch.tensor([0,1,0], device=self.__device)
                ) - self.solution_at(
                    positions[forward_diff_indices]
                ))*(1/self.__ydim)                      
            grads[backward_diff_indices, self.__channels:2*self.__channels] = \
                (-self.solution_at(
                    positions[backward_diff_indices]
                            - torch.tensor([0,1,0], device=self.__device)
                ) + self.solution_at(
                    positions[backward_diff_indices]
                ))*(1/self.__ydim)                  
            grads[central_diff_indices, self.__channels:2*self.__channels] = \
                (self.solution_at(
                    positions[central_diff_indices]
                            + torch.tensor([0,1,0], device=self.__device)
                ) - (self.solution_at(
                    positions[central_diff_indices]
                    - torch.tensor([0,1,0], device=self.__device))
                ))*(1/self.__ydim)  
                

            # x derivative
            forward_diff_indices : torch.Tensor = positions[:,2] == 0
            backward_diff_indices : torch.Tensor = positions[:,2] == self.__xdim-1
            central_diff_indices : torch.Tensor = torch.logical_and(
                positions[:,2] > 0,
                positions[:,2] < self.__xdim-1)
            
            grads[forward_diff_indices, 2*self.__channels:3*self.__channels] = \
                (self.solution_at(
                    positions[forward_diff_indices]
                            + torch.tensor([0,0,1], device=self.__device)
                ) - self.solution_at(
                    positions[forward_diff_indices]
                ))*(1/self.__xdim)                      
            grads[backward_diff_indices, 2*self.__channels:3*self.__channels] = \
                (-self.solution_at(
                    positions[backward_diff_indices]
                            - torch.tensor([0,0,1], device=self.__device)
                ) + self.solution_at(
                    positions[backward_diff_indices]
                ))*(1/self.__xdim)                  
            grads[central_diff_indices, 2*self.__channels:3*self.__channels] = \
                (self.solution_at(
                    positions[central_diff_indices]
                            + torch.tensor([0,0,1], device=self.__device)
                ) - (self.solution_at(
                    positions[central_diff_indices]
                    - torch.tensor([0,0,1], device=self.__device))
                ))*(1/self.__xdim)  

        return grads

    def interpolate(self, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        '''
        Computes the value (v) given an index (position) and 
        trilinear interpolation weights for the 8 neighboring cells. See
        Grid.get_cell_at() for information about weight orders.
        '''
        assert positions.dtype is torch.long
        values : torch.Tensor = torch.zeros([positions.shape[0], 
                                             self.__channels], 
                                            device=self.__device)
        
        max_values : torch.Tensor = torch.tensor(
            [self.__zdim-1, self.__ydim-1, self.__xdim-1], 
            device=self.__device).unsqueeze(0).repeat(
                positions.shape[0], 1
            )

        values += weights[:,0:1] * self.solution_at(positions)

        values += weights[:,1:2] * self.solution_at(torch.minimum(
            positions + torch.tensor([1, 0, 0], 
            device=self.__device),
            max_values))
        values += weights[:,2:3] * self.solution_at(torch.minimum(
            positions + torch.tensor([0, 1, 0], 
            device=self.__device),
            max_values))
        values += weights[:,3:4] * self.solution_at(torch.minimum(
            positions + torch.tensor([1, 1, 0], 
            device=self.__device),
            max_values))
        values += weights[:,4:5] * self.solution_at(torch.minimum(
            positions + torch.tensor([0, 0, 1], 
            device=self.__device),
            max_values))
        values += weights[:,5:6] * self.solution_at(torch.minimum(
            positions + torch.tensor([1, 0, 1], 
            device=self.__device),
            max_values))
        values += weights[:,6:7] * self.solution_at(torch.minimum(
            positions + torch.tensor([0, 1, 1], 
            device=self.__device),
            max_values))
        values += weights[:,7:8] * self.solution_at(torch.minimum(
            positions + torch.tensor([1, 1, 1], 
            device=self.__device),
            max_values))

        return values
    
    def interpolate_gradient(self, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        '''
        Computes the gradient (g) given an index (position) and 
        trilinear interpolation weights for the 8 neighboring cells. See
        Grid.get_cell_at() for information about weight orders.
        '''
        assert positions.dtype is torch.long

        values : torch.Tensor = torch.zeros([positions.shape[0], 
                                             3*self.__channels], device=self.__device)
        max_values = torch.tensor(
        [self.__zdim-1, self.__ydim-1, self.__xdim-1], 
        device=self.__device).unsqueeze(0).repeat(
            positions.shape[0], 1
        )
        
        values += weights[:,0:1] * self.grad_at(positions)
        values += weights[:,1:2] * self.grad_at(torch.minimum(
            positions + torch.tensor([1, 0, 0], 
            device=self.__device),
            max_values))
        values += weights[:,2:3] * self.grad_at(torch.minimum(
            positions + torch.tensor([0, 1, 0], 
            device=self.__device),
            max_values))
        values += weights[:,3:4] * self.grad_at(torch.minimum(
            positions + torch.tensor([1, 1, 0], 
            device=self.__device),
            max_values))
        values += weights[:,4:5] * self.grad_at(torch.minimum(
            positions + torch.tensor([0, 0, 1], 
            device=self.__device),
            max_values))
        values += weights[:,5:6] * self.grad_at(torch.minimum(
            positions + torch.tensor([1, 0, 1], 
            device=self.__device),
            max_values))
        values += weights[:,6:7] * self.grad_at(torch.minimum(
            positions + torch.tensor([0, 1, 1], 
            device=self.__device),
            max_values))
        values += weights[:,7:8] * self.grad_at(torch.minimum(
            positions + torch.tensor([1, 1, 1], 
            device=self.__device),
            max_values))

        return values

class Field(torch.nn.Module):
    '''
    A class that holds a solution and a grid associated with
    the solution. Can use this grid to query physical locations
    and return scalar values, or their gradients.
    '''

    def __init__(self, g : Grid, s : Solution):
        '''
        Initializes a Field with a grid and solution
        '''
        super().__init__()
        self.__grid : Grid = g
        self.__solution : Solution = s

    def set_data(self, data : torch.Tensor):
        self.__solution.set_data(data)

    def set_grid(self, g: Grid):
        '''
        Sets the grid for the field.
        '''
        self.__grid = g
    
    def get_min(self) -> float:
        return self.__solution.get_min()

    def get_max(self) -> float:
        return self.__solution.get_max()

    def device(self) -> str:
        return self.__solution.device()

    def to_device(self, device: str):
        self.__solution.to_device(device)

    def get_grid(self) -> Grid:
        return self.__grid

    def set_solution(self, s: Solution):
        '''
        Sets the solution for the field.
        '''
        self.__solution = s
    
    def get_solution(self) -> Solution:
        return self.__solution

    def get_extents(self) -> List[float]:
        '''
        Returns the extents (min/max) for each dimension
        in z, y, x order
        '''
        return self.__grid.get_extents()

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Computs the solutions at physical positions using the 
        private grid to find the correct indices, and the solution 
        class to solve the trilinear interpolation
        '''
        cells, weights = self.__grid.index_and_weights_for(positions)
        vals = self.__solution.interpolate(cells, weights)
        return vals

    def gradient_at_phys_pos(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Computs the gradient at a physical position using the 
        private grid to find the correct index, and the solution 
        class to solve the trilinear interpolation
        '''
        assert self.__grid.in_bounds(positions).all(), "Position must be in bounds"

        cells, weights = self.__grid.index_and_weights_for(positions)
        vals = self.__solution.interpolate_gradient(cells, weights)
        return vals

class TransferFunction():
    '''
    A class for transfer functions.
    '''

    def __init__(self, device: str):
        '''
        Initializes a Field with a grid and solution
        '''
        self.__device : str = device
        self.__min : float = 0.0
        self.__max : float = 1.0
        self.__normalized_rgb_positions : torch.Tensor = torch.tensor([0], device=self.__device)
        self.__normalized_rgb : torch.Tensor = torch.tensor([0], device=self.__device)

        self.__normalized_a_positions : torch.Tensor = torch.tensor([0], device=self.__device)
        self.__normalized_a : torch.Tensor = torch.tensor([0], device=self.__device)

    def set_min(self, m : float):
        self.__min : float = m
    
    def set_max(self, m : float):
        self.__max : float = m
    
    def set_colors_at_positions(self, cs: torch.Tensor, p: torch.Tensor):
        self.__normalized_rgb : torch.Tensor = cs.clone().to(self.__device)
        self.__normalized_rgb_positions : torch.Tensor = p.clone().to(self.__device)
        indices = torch.argsort(self.__normalized_rgb_positions)
        
        self.__normalized_rgb_positions : torch.Tensor = torch.index_select(
            self.__normalized_rgb_positions, 0, indices)
        self.__normalized_rgb : torch.Tensor = torch.index_select(
            self.__normalized_rgb, 0, indices)
    
    def set_alphas_at_positions(self, alphas: torch.Tensor, p: torch.Tensor):
        self.__normalized_a: torch.Tensor = alphas.clone().to(self.__device)
        self.__normalized_a_positions: torch.Tensor = p.clone().to(self.__device)
        indices = torch.argsort(self.__normalized_a_positions)
        
        self.__normalized_a_positions: torch.Tensor = torch.index_select(
            self.__normalized_a_positions, 0, indices)
        self.__normalized_a: torch.Tensor = torch.index_select(
            self.__normalized_a, 0, indices)
      
    def rgb_at_normalized_value(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clone().repeat([1, self.__normalized_rgb_positions.shape[0]-1])
        v[:] -= self.__normalized_rgb_positions[0:-1]
        v[:] /= (self.__normalized_rgb_positions[1:] \
            - self.__normalized_rgb_positions[0:-1])
        c_i = 1-v
        c_next = v
        c_i = torch.where(c_i >= 1.0, 
            torch.zeros_like(c_i),   c_i)
        c_next = torch.where(c_next > 1.0, 
            torch.zeros_like(c_next),   c_next)
        c_i.clamp_(0,1.0)
        c_next.clamp_(0,1.0)
        
        c = torch.zeros([v.shape[0], 3], 
            device=v.device)
        
        c += c_i @ self.__normalized_rgb[0:-1,:]        
        c += c_next @ self.__normalized_rgb[1:,:]

        return c
       
    def a_at_normalized_value(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clone().repeat([1, self.__normalized_a_positions.shape[0]-1])
        v[:] -= self.__normalized_a_positions[0:-1]
        v[:] /= (self.__normalized_a_positions[1:] \
            - self.__normalized_a_positions[0:-1])
        a_i = 1-v
        a_next = v
        a_i = torch.where(a_i >= 1.0, 
            torch.zeros_like(a_i),   a_i)
        a_next = torch.where(a_next > 1.0, 
            torch.zeros_like(a_next), a_next)
        a_i.clamp_(0,1.0)
        a_next.clamp_(0,1.0)
        
        a = torch.zeros([v.shape[0], 1], 
            device=v.device)
        
        a += a_i @ self.__normalized_a[0:-1,:]        
        a += a_next @ self.__normalized_a[1:,:]

        return a
    
    def rgba_at_normalized_value(self, v: torch.Tensor) -> torch.Tensor:
        rgb : torch.Tensor= self.rgb_at_normalized_value(v)
        a : torch.Tensor= self.a_at_normalized_value(v)    
            
        rgba : torch.Tensor= torch.cat([rgb, a], dim=1)
        return rgba

    def rgba_at_value(self, v: torch.Tensor) -> torch.Tensor:
        v -= self.__min
        v /= (self.__max - self.__min)
        return self.rgba_at_normalized_value(v)

def nvr_on_axis(model, dataset, tf, 
    axis="x", resolution=(128,128), 
    total_steps=128, phong_illumination=False,
    light_position = [-500.0, 0.0, 0.0],
    background = [1.0, 1.0, 1.0],
    device="cuda"):

    c_in : torch.Tensor= torch.zeros([resolution[0], resolution[1], 3], 
        device=device)
    c_out: torch.Tensor = torch.zeros([resolution[0], resolution[1], 3], 
        device=device)
    a_in: torch.Tensor = torch.zeros([resolution[0], resolution[1], 1], 
        device=device)
    a_out: torch.Tensor = torch.zeros([resolution[0], resolution[1], 1], 
        device=device)

    zyx: torch.Tensor = torch.tensor([0])
    delta : torch.Tensor = torch.tensor([0.0, 0.0, 0.0], 
            device=device)

    zyx_: List[torch.Tensor] = torch.meshgrid(
            [torch.linspace(-1, 1, steps=resolution[0]),
            torch.linspace(-1, 1, steps=resolution[1])],
            indexing='ij'
            )
    if(axis == "x"):
        
        zyx_ : List[torch.Tensor]= [zyx_[0], zyx_[1], torch.zeros_like(zyx_[0])-1]
        zyx = torch.stack(zyx_).type(torch.float32).to(device)
        
        delta[2] = 2 / (total_steps+1)

    elif(axis == "y"):

        zyx_: List[torch.Tensor] = [zyx_[0], torch.zeros_like(zyx_[0])-1, zyx_[1]]
        zyx = torch.stack(zyx_).type(torch.float32).to(device)
        
        delta[1] = 2 / (total_steps+1)

    elif(axis == "z"):

        zyx_: List[torch.Tensor] = [torch.zeros_like(zyx_[0])-1, zyx_[0], zyx_[1]]
        zyx = torch.stack(zyx_).type(torch.float32).to(device)
              
        delta[0] = 2 / (total_steps+1)

    p: torch.Tensor = zyx.permute(1, 2, 0)
    print("Beginning render")
    time_start: float = time.time()
    #solution_timing: float = 0.0
    #transfer_function_timing: float = 0.0
    #compositing_timing: float = 0.0
    light_position: torch.Tensor = torch.tensor(light_position, 
        device=device)
    start = -1
    end_dim = 1
    step_size = 2 / total_steps
    while(start < end_dim):
        p += delta
        start += step_size
        
        #t_0: float = time.time()
        flattened_p = p.flatten(0,1)
        #v = model(flattened_p)
        v = forward_maxpoints(model, flattened_p, 512**2)
        #solution_timing += (time.time() - t_0)
        
        #t_0: float = time.time()
        rgba = tf.rgba_at_value(v)
        #transfer_function_timing += (time.time() - t_0)
        
        
        rgb = rgba[:,0:3]
        a = rgba[:,3:4]
        if(phong_illumination):
            n = forward_w_grad(model,flattened_p)
            n /= (n.norm(dim=1, keepdim=True) + 1e-6)
            l = light_position - flattened_p
            l /= (l.norm(dim=1, keepdim=True) + 1e-6)

            angle: torch.Tensor = torch.bmm(n.unsqueeze(1),
                                            l.unsqueeze(2))[:,:,0]

            rgb[angle[:,0]>0] *= angle[angle[:,0]>0]

        #t_0 : float= time.time()

        
        rgb = rgb.reshape([p.shape[0], p.shape[1], 3])
        a = a.reshape([p.shape[0], p.shape[1], 1])

        c_out = c_in + rgb*a*(1-a_in)
        a_out = a_in + a*(1-a_in)
        
        c_in = c_out
        a_in = a_out
        #compositing_timing += (time.time() - t_0)
    
    print(f"Volume rendering took {time.time() - time_start} seconds.")
    if("cuda" in device):
        GBytes = (torch.cuda.max_memory_allocated(device=device) \
            / (1024**3))
        print(f"Max memory allocated {GBytes : 0.02f}GB")
    
    #print(f"Interpolation time: {solution_timing} seconds.")
    #print(f"Transfer function time: {transfer_function_timing} seconds.")
    #print(f"Compositing time: {compositing_timing} seconds.")
    background = torch.tensor(background, device=device)
    c_out = c_out + background*(1-a_out)
    return c_out, a_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs neural volume rendering')

    parser.add_argument('--load_from',default=None,type=str,
        help='Number of dimensions in the data')
    parser.add_argument('--device',default=None,type=str,
        help='Device to use.')

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    torch.manual_seed(11235813)

    opt = load_options(os.path.join(save_folder, args["load_from"]))
    opt["device"] = args["device"]
    opt['data_device'] = args['device']
    opt["save_name"] = args["load_from"]
    for k in args.keys():
        if args[k] is not None:
            opt[k] = args[k]
    dataset = get_dataset(opt)
    model = load_model(opt, opt['device'])


    tf = TransferFunction(args['device'])
    tf.set_colors_at_positions(
        torch.tensor([
            [0.23, 0.30, 0.75],
            [0.87, 0.87, 0.87],
            [0.70, 0.01, 0.15],
            ]),
        torch.tensor([0.0, 0.5, 1.0])
    )
    
    tf.set_alphas_at_positions(
        torch.tensor([
            0.0,
            0.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
            ]).unsqueeze(1),
        torch.tensor([
                    0.0,
                    0.2,
                    0.21,
                    0.22,
                    0.7,
                    0.71,
                    0.72,
                    1.0])
    )
    
    tf.set_min(-0.1)
    tf.set_max(0.16)
    
    
    with torch.no_grad():
        c_out, a_out = nvr_on_axis(model, dataset, tf,
                          resolution=[1024, 1024],
                          total_steps=4096,
                          axis='y',
                          device=args['device'])

    tensor_to_img(c_out, "./neural_render.jpg")
    

    #sampled_sf = nc_to_tensor(os.path.join(output_folder, "StreamFunction", opt['save_name']+".nc"))
    grid_size = [64, 64, 64]
    with torch.no_grad():
        sampled_sf = sample_grid(model, 
                                grid_size, 
                                max_points=10000)[...,0:1]
            
    g = Grid(
        -1, 1, grid_size[0],
        -1, 1, grid_size[1],
        -1, 1, grid_size[2],
    )
    s = Solution(
        3, grid_size[0], grid_size[1], grid_size[2],
        sampled_sf.to(opt['device']).squeeze().permute(2,1,0).unsqueeze(-1), opt['device']
    )
    
    f = Field(g, s)
    with torch.no_grad():
        c_out, a_out = nvr_on_axis(f, dataset, tf,
                          resolution=[1024, 1024],
                          total_steps=4096,
                          axis='y',
                          device=args['device'])

    background_color = torch.tensor([1.0, 1.0, 1.0, 1.0], device = c_out.device)
    tensor_to_img(c_out, "./render_on_grid.jpg")