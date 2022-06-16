import os
import torch
import netCDF4 as nc
from Other.utility_functions import make_coord_grid, normal
import torch.nn.functional as F
import numpy as np

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.full_coord_grid = None
        folder_to_load = os.path.join(data_folder, self.opt['data'])

        print(f"Initializing dataset - reading {folder_to_load}")
        
        f = nc.Dataset(folder_to_load)
        channels = []
        for a in f.variables:
            d = np.array(f[a])
            channels.append(d)
        d = np.stack(channels)
        d = torch.tensor(d).unsqueeze(0)
        d = d.to(opt['data_device'])   
        d /= (d.norm(dim=1).max() + 1e-8)
        self.data = d
            
        if("direction" in opt['training_mode'] or "parallel" in opt['training_mode']):
            n = normal(d, normalize=True)
            self.normal = n
        self.index_grid = make_coord_grid(
            self.data.shape[2:], 
            self.opt['data_device'],
            flatten=True)
        self.index_mags = self.index_grid.norm(dim=1)
        print("Data size: " + str(self.data.shape))
        print("Min/mean/max: %0.04f, %0.04f, %0.04f" % \
            (self.min(), self.mean(), self.max()))
        print("Min/mean/max mag: %0.04f, %0.04f, %0.04f" % \
            (self.data.norm(dim=1).min(), 
            self.data.norm(dim=1).mean(), 
            self.data.norm(dim=1).max()))

    def min(self):
        if self.min_ is not None:
            return self.min_
        else:
            self.min_ = self.data.min()
            return self.min_
    def mean(self):
        if self.mean_ is not None:
            return self.mean_
        else:
            self.mean_ = self.data.mean()
            return self.mean_
    def max(self):
        if self.max_ is not None:
            return self.max_
        else:
            self.max_ = self.data.max()
            return self.max_

    def get_2D_slice(self):
        if(len(self.data.shape) == 4):
            return self.data[0].clone()
        else:
            return self.data[0,:,:,:,int(self.data.shape[4]/2)].clone()

    def sample_rect(self, starts, widths, samples):
        positions = []
        for i in range(len(starts)):
            positions.append(
                torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                    dtype=torch.float32, device=self.opt['data_device'])
            )
            positions[i] -= 0.5
            positions[i] *= 2
        grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1).unsqueeze(0)

        vals = F.grid_sample(self.data, 
                grid_to_sample, mode='bilinear', align_corners=False)
        print('dataset sample rect vals shape')
        print(vals.shape)
        return vals

    def total_points(self):
        t = 1
        for i in range(2, len(self.data.shape)):
            t *= self.data.shape[i]
        return t

    def get_full_coord_grid(self):
        if self.full_coord_grid is None:
            self.full_coord_grid = make_coord_grid(self.data.shape[2:], 
                    self.opt['data_device'], flatten=True).unsqueeze(0)
        return self.full_coord_grid

    def get_random_points(self, n_points):        
        possible_spots = self.index_grid

        if(n_points >= possible_spots.shape[0]):
            x = possible_spots.clone().unsqueeze_(0)
        else:
            samples = torch.randperm(possible_spots.shape[0], 
                dtype=torch.long, device=self.opt['data_device'])[:n_points]
            # Change above to not use CPU when not on MPS
            # Verify that the bottom two lines do the same thing
            x = torch.index_select(possible_spots, 0, samples).clone().unsqueeze_(0)
            #x = possible_spots[samples].clone().unsqueeze_(0)
        for _ in range(len(self.data.shape[2:])-1):
            x = x.unsqueeze(-2)
        

        y = F.grid_sample(self.data, 
            x, mode='nearest', align_corners=False)
        
        y = y.squeeze()
        if(len(y.shape) == 1):
            y = y.unsqueeze(0)    
        
        y = y.permute(1,0)
            
        if('parallel' in self.opt['training_mode'] or 
           'direction' in self.opt['training_mode']):
            y_n = F.grid_sample(self.normal, 
                x, mode='nearest', align_corners=False)
            y_n = y_n.squeeze()            
            if(len(y_n.shape) == 1):
                y_n = y_n.unsqueeze(0)  
            y_n = y_n.permute(1,0)
                
        x = x.squeeze()

        to_return = {
            "inputs": x,
            "data": y
        }
        if('parallel' in self.opt['training_mode'] or 
           'direction' in self.opt['training_mode']):
            to_return["normal"] = y_n
        
        return to_return
