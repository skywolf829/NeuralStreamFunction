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
from nn_compression.prune import VanillaPruner
from nn_compression.quantize import Quantizer

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def l1(x, y):
    return F.l1_loss(x, y)

def mse(x, y):
    return F.mse_loss(x, y)

def l1_occupancy(gt, y):
    # Expects x to be [..., 3] or [..., 4] for (u, v, o) or (u, v, w, o)
    # Where o is occupancy
    is_nan_mask = torch.isnan(gt)[...,0].detach()
    
    o_loss = l1((~is_nan_mask).to(torch.float32).detach(), y[..., -1])
    vf_loss = l1(gt[~is_nan_mask, :].detach(), y[~is_nan_mask, 0:-1])
    return o_loss + vf_loss

def perpendicular_loss(x, y):
    return F.cosine_similarity(x, y).mean()

def angle_same_loss(x, y):
    angles = (1 - (F.cosine_similarity(x, y)**2)).mean()
    return angles

def angle_orthogonal_loss(x, y):
    angles = (F.cosine_similarity(x, y)**2).mean()
    return angles

def magangle_orthogonal_loss(x, y):
    mags = F.l1_loss(torch.norm(x,dim=1), torch.norm(y,dim=1))
    angles = (F.cosine_similarity(x, y)**2).mean()
    return mags + angles

def magangle_same_loss(x, y):
    mags = F.l1_loss(torch.norm(x,dim=1), torch.norm(y,dim=1))
    angles = (1 - (F.cosine_similarity(x, y)**2)).mean()
    return 0.9*mags + 0.1*angles

def train_loop(model, dataset, loss_func, opt):
    model.zero_grad()
    x, y = dataset.get_random_points(opt['points_per_iteration'])
    #print(y.isnan().any().sum() / y.shape[0])
    x = x.to(opt['device'])
    y = y.to(opt['device'])
    
    if(opt['fit_gradient']):
        y_estimated, x = model.forward_w_grad(x)
        y_estimated = torch.autograd.grad(y_estimated, x, 
                grad_outputs=torch.ones_like(y_estimated),
                create_graph=True)[0]
    elif(opt['dual_streamfunction']):
        y_estimated, x = model.forward_w_grad(x)
        grads_f = torch.autograd.grad(y_estimated[:,0], x, 
                grad_outputs=torch.ones_like(y_estimated[:,0]),
                create_graph=True)[0]
        grads_g = torch.autograd.grad(y_estimated[:,1], x, 
                grad_outputs=torch.ones_like(y_estimated[:,1]),
                create_graph=True)[0]
        y_estimated = torch.cross(grads_f, grads_g, dim=1)
    else:
        y_estimated = model(x)
    loss = loss_func(y, y_estimated)
    loss.backward()

    return x, y, y_estimated, loss

def log_to_writer(iteration, y, y_estimated, loss, writer, dataset, opt):    
    with torch.no_grad():
        #p_vf = PSNR(y_estimated.detach(), y.detach(), 
        #    dataset.max()-dataset.min())
        #writer.add_scalar('PSNR', p_vf.item(), iteration)
        #writer.add_scalar('loss', loss.item(), iteration)
        print("Iteration %i/%i, loss: %0.06f" % \
                (iteration, opt['iterations'], loss.item()))
        
        if 'same' in opt['loss'] or "l1" in opt['loss']:
            p = PSNR(y, y_estimated, range=y.max()-y.min())
            print(f"PSNR: {p : 0.02f}")

        writer.add_scalar('Loss', loss.item(), iteration)
        GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
            / (1024**3))
        writer.add_scalar('GPU memory (GB)', GBytes, iteration)

def log_image(model, dataset, grid_to_sample, writer, iteration):
    with torch.no_grad():
        img = model.sample_grid_for_image(grid_to_sample)
        print("img shape: " + str(img.shape))
        if(dataset.min() < 0 or dataset.max() > 1.0):
            img -= dataset.min()
            img /= (dataset.max() - dataset.min())
        writer.add_image('Reconstruction', img.clamp(0, 1), 
            iteration, dataformats='HWC')

def log_grad_image(model, grid_to_sample, writer, iteration):
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
                iteration, dataformats='HWC')

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

    
    
    if(opt['train_distributed']):
        model = DDP(model, device_ids=[rank])
    else:
        model = model.to(opt['device'])
        
    print("Training on %s" % (opt["device"]), 
        os.path.join(save_folder, opt["save_name"]))


    optimizer = optim.Adam(model.parameters(), lr=opt["lr"],
        betas=[opt['beta_1'], opt['beta_2']]) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=3333, gamma=0.1)

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
    elif(opt['loss'] == 'magangle_same'):
        loss_func = magangle_same_loss    
    elif(opt['loss'] == 'magangle_orthogonal'):
        loss_func = magangle_orthogonal_loss        
    elif(opt['loss'] == 'angle_same'):
        loss_func = angle_same_loss
    elif(opt['loss'] == 'angle_orthogonal'):
        loss_func = angle_orthogonal_loss
    elif(opt['loss'] == 'mse'):
        loss_func = mse
    model.train(True)


    for iteration in range(0, opt['iterations']):
        
        x, y, y_estimated, loss = train_loop(model, dataset, loss_func, opt)

        optimizer.step()
        scheduler.step()
        if((rank == 0 and opt['train_distributed']) or not opt['train_distributed']):
            if(iteration % opt['save_every'] == 0):
                save_model(model, opt)

            if(iteration % 5 == 0):
                log_to_writer(iteration, y, y_estimated, loss, writer, dataset, opt)
                    
            
            if(iteration % 100 == 0 and (opt['log_image'] or opt['log_gradient'])):
                grid_to_sample = dataset.data.shape[2:]
                if(opt['log_image']):
                    log_image(model, dataset, grid_to_sample, writer, iteration)
                if(opt['log_gradient']):
                    log_grad_image(model, grid_to_sample, writer, iteration)
    
    

    if(opt['pruning'] != None):
        print("pruning")
        pruning_rules = []
        for name, param in model.named_parameters():
            print(name, param.size())
            pruning_rules.append(
                [name, 'element', 
                    opt['pruning'], 'abs']
            )
        pruner = VanillaPruner(rule=pruning_rules)

        for epoch in range(0, 100):
            if(epoch == 0):
                pruner.prune(model, update_masks=True)
            x, y, y_estimated, loss = train_loop(model, dataset, loss_func, opt)
            optimizer.step()
            scheduler.step()
            pruner.prune(model, update_masks=False)
            if(epoch % 5 == 0):
                log_to_writer(iteration, y, y_estimated, loss, writer, dataset, opt)
    
    if(opt['quantization'] != None):
        print("quantizing")
        quantization_rules = []
        for name, param in model.named_parameters():
            print(name, param.size())
            quantization_rules.append(
                [name, 'k-means', 
                    opt['quantization'], 'k-means++']
                )
        quantizer = Quantizer(rule=quantization_rules, fix_zeros=True)
        
        for epoch in range(0, 100):
            x, y, y_estimated, loss = train_loop(model, dataset, loss_func, opt)
            optimizer.step()
            scheduler.step()
            quantizer.quantize(model=model, update_labels=True, re_quantize=False)
            if(epoch % 5 == 0):
                log_to_writer(iteration, y, y_estimated, loss, writer, dataset, opt)


    if((rank == 0 and opt['train_distributed']) or not opt['train_distributed']):
        writer.close()

    save_model(model, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--n_dims',default=None,type=int)
    parser.add_argument('--n_outputs',default=None,type=int)
    parser.add_argument('--activation_function',default=None,type=str)
    parser.add_argument('--periodic',default=None,type=str2bool)
    parser.add_argument('--use_positional_encoding',default=None,type=str2bool)
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int)
    parser.add_argument('--dropout',default=None,type=str2bool)
    parser.add_argument('--dropout_p',default=None,type=float)
    parser.add_argument('--interpolate',default=None,type=str2bool)
    parser.add_argument('--fit_gradient',default=None,type=str2bool)
    parser.add_argument('--dual_streamfunction',default=None,type=str2bool)
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
    parser.add_argument('--pruning',default=None, type=float)
    parser.add_argument('--quantization',default=None, type=int)
    parser.add_argument('--coding',default=None, type=str2bool)


    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    torch.manual_seed(11235813)

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

        

