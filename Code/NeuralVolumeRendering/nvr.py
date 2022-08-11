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
from models import load_model, forward_w_grad
from datasets import get_dataset
import time
from typing import List
import imageio
import numpy as np

def tensor_to_img(t: torch.Tensor, path:str):    
    img_data: np.ndarray = t.detach().cpu().numpy()
    img_data *= 255
    img_data = img_data.astype(np.uint8)
    imageio.imwrite(path, img_data)

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
        v = model(flattened_p)
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
    #print(f"Interpolation time: {solution_timing} seconds.")
    #print(f"Transfer function time: {transfer_function_timing} seconds.")
    #print(f"Compositing time: {compositing_timing} seconds.")
    
    return c_out

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
            0.1,
            0.0,
            0.0
            ]).unsqueeze(1),
        torch.tensor([0.0,
                    0.2,
                    0.21,
                    0.22,
                    0.7,
                    0.71,
                    0.72,
                    1.0])
    )
    
    tf.set_min(-0.1)
    tf.set_max(0.17)
    
    with torch.no_grad():
        c_out = nvr_on_axis(model, dataset, tf,
                          resolution=[512, 512],
                          total_steps=256,
                          axis='x',
                          device=args['device'])

    tensor_to_img(c_out, "./render.jpg")