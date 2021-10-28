import os
import torch
import h5py
from utility_functions import make_coord_grid
import torch.nn.functional as F

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")



class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.items = None
        self.item_names = None 
        self.coord_grid = None

        folder_to_load = os.path.join(data_folder, self.opt['vector_field_name'])

        print("Initializing dataset - reading %s" % folder_to_load)
        
        f = h5py.File(folder_to_load, 'r')
        d = torch.tensor(f.get('data'))
        f.close()
        self.data = d
        self.data = self.data.to(self.opt['data_device']).unsqueeze(0)

    def get_random_points(self, n_points):        
        if(self.opt['interpolate']):
            x = (torch.rand([n_points, len(self.data.shape[2:])], 
                device=self.data.device) - 1) * 2
        else:
            if(self.coord_grid is None):
                self.coord_grid = make_coord_grid(self.data.shape[2:], 
                    self.opt['data_device'], True)
            sample_spots = torch.randint(0, self.coord_grid.shape[0], 
                [self.coord_grid.shape[0]], device=self.opt['data_device'])
            x = self.coord_grid[sample_spots].clone()
        
        y = F.grid_sample(self.data, 
                x, mode='bilinear', align_corners=False)
            
        return x, y
