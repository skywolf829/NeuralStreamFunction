from __future__ import absolute_import, division, print_function
import argparse
from datasets import Dataset
import datetime
from utility_functions import str2bool, PSNR
from models import load_model, save_model, ImplicitModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
from options import *
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def l1(x, y):
    return F.l1_loss(x, y)

def l1_occupancy(x, y):
    # Expects x to be [..., 3] or [..., 4] for (u, v, o) or (u, v, w, o)
    # Where o is occupancy
    is_nan_mask = torch.isnan(x)[...,0]
    
    o_loss = l1(is_nan_mask.to(torch.float32), y[..., -1])
    vf_loss = l1(x[~is_nan_mask, :], y[~is_nan_mask, 0:-1])
    return o_loss + vf_loss

def perpendicular_loss(x, y):
    return F.cosine_similarity(x, y).mean()


def train_implicit_model(rank, model, dataset, opt):
    print("Training on device " + str(rank))
    if(opt['train_distributed']):        
        print("Initializing process group.")
        opt['device'] = "cuda:" + str(rank)
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=opt['gpus_per_node'],                              
            rank=rank                                               
        )  
    start_t = time.time()

    torch.manual_seed(0)
    
    
    if(opt['train_distributed']):
        model = DDP(model, device_ids=[rank])
    else:
        model = model.to(opt['device'])
        
    print("Training on %s" % (opt["device"]), 
        os.path.join(save_folder, opt["save_name"]))


    optimizer = optim.Adam(model.parameters(), lr=opt["lr"],
        betas=[opt['beta_1'], opt['beta_2']]) 

    if((rank == 0 and opt['train_distributed']) or not opt['train_distributed']):
        writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
        gt_img = dataset.get_2D_slice()
        gt_img -= dataset.min()
        gt_img /= (dataset.max() - dataset.min())
        writer.add_image("Ground Truth", gt_img, 0, dataformats="CHW")
    
       
    if(opt['loss'] == 'l1'):
        loss_func = l1
    elif(opt['loss'] == "perpendicular"):
        loss_func = perpendicular_loss
    elif(opt['loss'] == 'l1occupancy'):
        loss_func = l1_occupancy

    for iteration in range(0, opt['iterations']):
        model.zero_grad()
        x, y = dataset.get_random_points(opt['points_per_iteration'])
        x = x.to(opt['device'])
        y = y.to(opt['device'])

        y_estimated = model(x)
        loss = loss_func(y, y_estimated)
        loss.backward()

        optimizer.step()

        if((rank == 0 and opt['train_distributed']) or not opt['train_distributed']):
            if(iteration % opt['save_every'] == 0):
                save_model(model, opt)

            if(iteration % 5 == 0):
                with torch.no_grad():
                    print("Iteration %i/%i, loss: %0.06f" % \
                            (iteration, opt['iterations'], 
                            loss.item()))
                    writer.add_scalar('Loss', loss.item(), iteration)

                    GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
                        / (1024**3))
                    writer.add_scalar('GPU memory (GB)', GBytes, iteration)
            
                    if(opt['loss'] == "l1occupancy"):
                        print(y_estimated.shape)
                        print(y.shape)
                        print(y_estimated.detach()[:,0:-1].shape)
                        print(y_estimated.detach()[:, -1].shape)
                        print(torch.isnan(y[:,0]).to(torch.float32).shape)
                        p_vf = PSNR(y_estimated.detach()[:,0:-1], y.detach(), 
                            dataset.max()-dataset.min())
                        p_occupancy = PSNR(y_estimated.detach()[:, -1], torch.isnan(y[:,0]).to(torch.float32))
                        writer.add_scalar('PSNR', p_vf.item(), iteration)
                        writer.add_scalar('PSNR_occupancy', p_occupancy.item(), iteration)
                    else:
                        p_vf = PSNR(y_estimated.detach(), y.detach(), 
                            dataset.max()-dataset.min())
                        writer.add_scalar('PSNR', p_vf.item(), iteration)
                    
            
            if(iteration % 100 == 0 and (opt['log_image'] or opt['log_gradient'])):
                grid_to_sample = dataset.data.shape[2:]
                if(opt['log_image']):
                    with torch.no_grad():
                        img = model.sample_grid_for_image(grid_to_sample)
                        if(dataset.min() < 0 or dataset.max() > 1.0):
                            img -= dataset.min()
                            img /= (dataset.max() - dataset.min())
                        writer.add_image('Reconstruction', img.clamp(0, 1), 
                            iteration, dataformats='WHC')
                if(opt['log_gradient']):
                    grad_img = model.sample_grad_grid_for_image(grid_to_sample)
                    for output_index in range(len(grad_img)):
                        for input_index in range(grad_img[output_index].shape[-1]):
                            grad_img[output_index][...,input_index] -= \
                                grad_img[output_index][...,input_index].min()
                            grad_img[output_index][...,input_index] /= \
                                grad_img[output_index][...,input_index].max()

                            writer.add_image('Gradient_outputdim'+str(output_index)+\
                                "_wrt_inpudim_"+str(input_index), 
                                grad_img[output_index][...,input_index:input_index+1].clamp(0, 1), 
                                iteration, dataformats='WHC')
    
    if((rank == 0 and opt['train_distributed']) or not opt['train_distributed']):
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--n_dims',default=None,type=int)
    parser.add_argument('--n_outputs',default=None,type=int)
    parser.add_argument('--activation_function',default=None,type=str)
    parser.add_argument('--periodic',default=None,type=str2bool)
    parser.add_argument('--use_positional_encoding',default=None,type=str2bool)
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int)
    parser.add_argument('--interpolate',default=None,type=str2bool)
    parser.add_argument('--signal_file_name',default=None,type=str)
    parser.add_argument('--save_name',default=None,type=str)
    parser.add_argument('--n_layers',default=None,type=int)
    parser.add_argument('--nodes_per_layer',default=None,type=int)    
    parser.add_argument('--loss',default=None,type=str)
    parser.add_argument('--train_distributed',default=None,type=str2bool)
    parser.add_argument('--device',default=None, type=str)
    parser.add_argument('--data_device',default=None,type=str)
    parser.add_argument('--gpus_per_node',default=None, type=int)
    parser.add_argument('--num_nodes',default=None, type=int)
    parser.add_argument('--ranking',default=None, type=int)
    parser.add_argument('--iterations',default=None, type=int)
    parser.add_argument('--points_per_iteration',default=None, type=int)
    parser.add_argument('--lr',default=None, type=float)
    parser.add_argument('--beta_1',default=None, type=float)
    parser.add_argument('--beta_2',default=None, type=float)
    parser.add_argument('--iteration_number',default=None, type=int)
    parser.add_argument('--save_every',default=None, type=int)
    parser.add_argument('--log_every',default=None, type=int)
    parser.add_argument('--load_from',default=None, type=str)
    parser.add_argument('--log_image',default=None, type=str2bool)
    parser.add_argument('--log_gradient',default=None, type=str2bool)


    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")


    if(args['load_from'] is None):
        # Init models
        model = None
        opt = Options.get_default()

        # Read arguments and update our options
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]

        # Determine scales    
        dataset = Dataset(opt)
        model = ImplicitModel(opt)
    else:        
        opt = load_options(os.path.join(save_folder, args["load_from"]))
        opt["device"] = args["device"]
        opt["save_name"] = args["load_from"]
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        dataset = Dataset(opt)
        model = load_model(opt, opt['device'])

    now = datetime.datetime.now()
    start_time = time.time()
    
        

       
    if(opt['train_distributed']):
        os.environ['MASTER_ADDR'] = '127.0.0.1'              
        os.environ['MASTER_PORT'] = '29500' 
        mp.spawn(train_implicit_model,
            args=(model, dataset, opt),
            nprocs=opt['gpus_per_node'],
            join=True)
    else:
        train_implicit_model(opt['device'], model, 
                dataset,opt)
        
    opt['iteration_number'] = 0

        

