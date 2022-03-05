import os
import torch
import h5py
from utility_functions import make_coord_grid, normal, binormal
import torch.nn.functional as F
import numpy as np

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
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
        
        folder_to_load = os.path.join(data_folder, self.opt['signal_file_name'])

        print("Initializing dataset - reading %s" % folder_to_load)
        
        f = h5py.File(folder_to_load, 'r')
        d = torch.tensor(np.array(f.get('data'))).unsqueeze(0).to(self.opt['data_device'])
        f.close()
        d /= (d.norm(dim=1).max() + 1e-8)
            
        if(opt['fit_gradient'] and opt['gradient_direction'] == "N"):
            print("calculating normal direction")
            d = normal(d, normalize=True)
        elif(opt['fit_gradient'] and opt['gradient_direction'] == "B"):           
            print("calculating binormal direction")
            d = binormal(d, normalize=True)
        elif(opt['dual_stream_function'] == "N"):
            print("Calculating N")
            self.n = normal(d, normalize=True)
            
        self.data = d
        self.index_grid = make_coord_grid(self.data.shape[2:], self.opt['data_device'])
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
        #print(grid_to_sample.shape)
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
        if(self.opt['interpolate']):
            x = (torch.rand([1, n_points, len(self.data.shape[2:])], 
                device=self.data.device) - 0.5) * 2
            for _ in range(len(self.data.shape[2:])-1):
                x = x.unsqueeze(-2)
            
            if(self.opt['dual_stream_function'] == "N"):
                y_n = F.grid_sample(self.n, 
                    x, mode='bilinear', align_corners=False)
                y = F.grid_sample(self.data, 
                    x, mode='bilinear', align_corners=False)
            else:
                y = F.grid_sample(self.data, 
                    x, mode='bilinear', align_corners=False)
        else:
            x_dims = []
            if(n_points >= self.total_points()):
                x = self.get_full_coord_grid()
            else:
                samples = torch.rand(self.index_grid.shape[0], 
                    dtype=torch.float32, device=self.opt['data_device']) < \
                        n_points / self.index_grid.shape[0]
                x = self.index_grid[samples].clone().unsqueeze_(0)
            for _ in range(len(self.data.shape[2:])-1):
                x = x.unsqueeze(-2)
            if(self.opt['dual_stream_function'] == "N"):
                y_n = F.grid_sample(self.n, 
                    x, mode='nearest', align_corners=False)
                y = F.grid_sample(self.data, 
                    x, mode='nearest', align_corners=False)
                y = torch.cat([y, y_n], dim=1)
            else:
                y = F.grid_sample(self.data, 
                    x, mode='nearest', align_corners=False)
                
        x = x.squeeze()
        y = y.squeeze()
        if(len(y.shape) == 1):
            y = y.unsqueeze(0)        
        y = y.permute(1,0)

        return x, y
