import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import load_SSRTVD_models
from options import  load_options
from utility_functions import  str2bool, AvgPool3D, AvgPool2D
import os
import argparse
import time
from math import log2
from datasets import TestingDataset
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy
from utility_functions import ssim, ssim3D, save_obj, load_obj


def mse_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    return ((GT-x)**2).mean()

def psnr_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    data_range = GT.max() - GT.min()
    return (20.0*torch.log10(data_range)-10.0*torch.log10(mse_func(GT, x, device)))

def mre_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    data_range = GT.max() - GT.min()
    return (torch.abs(GT-x).max() / data_range)

def generate_by_patch(generator, input_volume, patch_size, receptive_field, device):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2]
            ).to(device)
        
        rf = receptive_field
                    
        z_done = False
        z = 0
        z_stop = min(input_volume.shape[2], z + patch_size)
        while(not z_done):
            if(z_stop == input_volume.shape[2]):
                z_done = True
            y_done = False
            y = 0
            y_stop = min(input_volume.shape[3], y + patch_size)
            while(not y_done):
                if(y_stop == input_volume.shape[3]):
                    y_done = True
                x_done = False
                x = 0
                x_stop = min(input_volume.shape[4], x + patch_size)
                while(not x_done):                        
                    if(x_stop == input_volume.shape[4]):
                        x_done = True
                    print("%d:%d, %d:%d, %d:%d" % (z, z_stop, y, y_stop, x, x_stop))
                    result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])

                    x_offset = rf if x > 0 else 0
                    y_offset = rf if y > 0 else 0
                    z_offset = rf if z > 0 else 0

                    final_volume[:,:,
                    2*z+z_offset:2*z+result.shape[2],
                    2*y+y_offset:2*y+result.shape[3],
                    2*x+x_offset:2*x+result.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]

                    x += patch_size - 2*rf
                    x = min(x, max(0, input_volume.shape[4] - patch_size))
                    x_stop = min(input_volume.shape[4], x + patch_size)
                y += patch_size - 2*rf
                y = min(y, max(0, input_volume.shape[3] - patch_size))
                y_stop = min(input_volume.shape[3], y + patch_size)
            z += patch_size - 2*rf
            z = min(z, max(0, input_volume.shape[2] - patch_size))
            z_stop = min(input_volume.shape[2], z + patch_size)

    return final_volume

def generate_patch(z,z_stop,y,y_stop,x,x_stop,available_gpus):

    device = None
    while(device is None):        
        device, generator, input_volume = available_gpus.get_next_available()
        time.sleep(1)
    #print("Starting SR on device " + device)
    with torch.no_grad():
        result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])
    return result,z,z_stop,y,y_stop,x,x_stop,device

class SharedList(object):  
    def __init__(self, items, generators, input_volumes):
        self.lock = threading.Lock()
        self.list = items
        self.generators = generators
        self.input_volumes = input_volumes
        
    def get_next_available(self):
        #print("Waiting for a lock")
        self.lock.acquire()
        item = None
        generator = None
        input_volume = None
        try:
            #print('Acquired a lock, counter value: ', self.counter)
            if(len(self.list) > 0):                    
                item = self.list.pop(0)
                generator = self.generators[item]
                input_volume = self.input_volumes[item]
        finally:
            #print('Released a lock, counter value: ', self.counter)
            self.lock.release()
        return item, generator, input_volume
    
    def add(self, item):
        #print("Waiting for a lock")
        self.lock.acquire()
        try:
            #print('Acquired a lock, counter value: ', self.counter)
            self.list.append(item)
        finally:
            #print('Released a lock, counter value: ', self.counter)
            self.lock.release()

def generate_by_patch_parallel(generator, input_volume, patch_size, receptive_field, devices):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2]
            ).to(devices[0])
        
        rf = receptive_field

        available_gpus = []
        generators = {}
        input_volumes = {}

        for i in range(1, len(devices)):
            available_gpus.append(devices[i])
            g = copy.deepcopy(generator).to(devices[i])
            iv = input_volume.clone().to(devices[i])
            generators[devices[i]] = g
            input_volumes[devices[i]] = iv
            torch.cuda.empty_cache()

        available_gpus = SharedList(available_gpus, generators, input_volumes)

        threads= []
        with ThreadPoolExecutor(max_workers=len(devices)-1) as executor:
            z_done = False
            z = 0
            z_stop = min(input_volume.shape[2], z + patch_size)
            while(not z_done):
                if(z_stop == input_volume.shape[2]):
                    z_done = True
                y_done = False
                y = 0
                y_stop = min(input_volume.shape[3], y + patch_size)
                while(not y_done):
                    if(y_stop == input_volume.shape[3]):
                        y_done = True
                    x_done = False
                    x = 0
                    x_stop = min(input_volume.shape[4], x + patch_size)
                    while(not x_done):                        
                        if(x_stop == input_volume.shape[4]):
                            x_done = True
                        
                        
                        threads.append(
                            executor.submit(
                                generate_patch,
                                z,z_stop,
                                y,y_stop,
                                x,x_stop,
                                available_gpus
                            )
                        )
                        
                        x += patch_size - 2*rf
                        x = min(x, max(0, input_volume.shape[4] - patch_size))
                        x_stop = min(input_volume.shape[4], x + patch_size)
                    y += patch_size - 2*rf
                    y = min(y, max(0, input_volume.shape[3] - patch_size))
                    y_stop = min(input_volume.shape[3], y + patch_size)
                z += patch_size - 2*rf
                z = min(z, max(0, input_volume.shape[2] - patch_size))
                z_stop = min(input_volume.shape[2], z + patch_size)

            for task in as_completed(threads):
                result,z,z_stop,y,y_stop,x,x_stop,device = task.result()
                result = result.to(devices[0])
                x_offset_start = rf if x > 0 else 0
                y_offset_start = rf if y > 0 else 0
                z_offset_start = rf if z > 0 else 0
                x_offset_end = rf if x_stop < input_volume.shape[4] else 0
                y_offset_end = rf if y_stop < input_volume.shape[3] else 0
                z_offset_end = rf if z_stop < input_volume.shape[2] else 0
                #print("%d, %d, %d" % (z, y, x))
                final_volume[:,:,
                2*z+z_offset_start:2*z+result.shape[2] - z_offset_end,
                2*y+y_offset_start:2*y+result.shape[3] - y_offset_end,
                2*x+x_offset_start:2*x+result.shape[4] - x_offset_end] = result[:,:,
                z_offset_start:result.shape[2]-z_offset_end,
                y_offset_start:result.shape[3]-y_offset_end,
                x_offset_start:result.shape[4]-x_offset_end]
                available_gpus.add(device)
    
    return final_volume

def get_test_results(GT, x, mode):
    p = psnr_func(GT, x, GT.device).item()
    ms = mse_func(GT, x, GT.device).item()
    mr = mre_func(GT, x, GT.device).item()
    if(mode == "2D"):
        s = ssim(GT, x).item()
    else:
        s = ssim3D(GT, x).item()

    return {"PSNR (dB)": p, "SSIM": s, "MSE": ms, "MRE": mr}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    
    parser.add_argument('--mode',default="3D",type=str,help='2D or 3D')
    parser.add_argument('--data_folder',default="Isomag3D",type=str,help='Name of dataset to test')
    parser.add_argument('--model_name',default="Isomag3D",type=str,help='The folder with the model to load')
    parser.add_argument('--device',default="cuda:0",type=str,help='Device to use for testing')
    parser.add_argument('--dict_entry_name',default="model_SSRTVD",type=str,help='Name for model in dict')    
    parser.add_argument('--parallel',default="False",type=str2bool,help='Perform SR in parallel')
    parser.add_argument('--test_on_gpu',default="True",type=str2bool,help='Metrics calculated on GPU?')
    parser.add_argument('--output_file_name',default="Isomag3D.results",type=str,help='Where to write results')
    
    args = vars(parser.parse_args())


    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    print("Loading options and model")
    opt = load_options(os.path.join(save_folder, args["model_name"]))

    opt["device"] = args["device"]
    opt['data_folder'] = args['data_folder']

    generator, _, _ = load_SSRTVD_models(opt,"cpu")
    generator = generator.to(opt['device'])
    generator.train(False)

    dataset = TestingDataset(opt['data_folder'])
    results_location = os.path.join(output_folder, args['output_file_name'])

    if(torch.cuda.device_count() > 1 and args['parallel']):
        devices = []
        for i in range(torch.cuda.device_count()):
            devices.append("cuda:"+str(i))
    
    # Maps SR factor -> upscale mode -> results
    # I.e. "2x" -> "trilinear" -> "PSNR", etc
    all_results = {

    }
    if(os.path.exists(results_location)):
        all_results = load_obj(results_location)
    print(all_results)

    with torch.no_grad():
        
        scale_factor_in_testing = 4 + "x"
        this_scale_results = {
            args['dict_entry_name']: {
                "Upscaling time": [],
                "MSE": [],
                "PSNR (dB)": [],
                "SSIM": [],
                "MRE": []
            }
        }
        for i in range(len(dataset)):
            GT_data = dataset[i].clone().to(args['device']).unsqueeze(0)
            print("Data size: " + str(GT_data.shape))
                
            chans = []
            if(args['mode'] == "3D"):
                for j in range(GT_data.shape[1]):
                    LR_data = AvgPool3D(GT_data[:,j:j+1,:,:,:], 4)
                    chans.append(LR_data)
            elif(args['mode'] == "2D"):
                for j in range(GT_data.shape[1]):
                    LR_data = AvgPool2D(GT_data[:,j:j+1,:,:], 4)
                    chans.append(LR_data)
            LR_data = torch.cat(chans, dim=1)

            print("Finished downscaling to " + str(LR_data.shape) + ". Performing super resolution")
            
            inference_start_time = time.time()
            
            x = LR_data.clone()
            current_ds = 4            
                
            if(torch.cuda.device_count() > 1 and args['parallel'] and args['mode'] == '3D'):
                print("Upscaling in parallel on " + str(len(devices)) + " gpus")
                x = generate_by_patch_parallel(generator, 
                    x, 140, 10, devices)
            else:
                if(args['mode'] == '3D'):
                    x = generate_by_patch(generator, 
                        x, 140, 10, args['device'])
                elif(args['mode'] == '2D'):
                    x = generator(x)

            current_ds = int(current_ds / 2)
            inference_end_time = time.time()                
            inference_this_frame = inference_end_time - inference_start_time

            print("Finished super resolving in %0.04f seconds. Final shape: %s. Performing tests." % \
                (inference_this_frame, str(LR_data.shape)))
            frame_results = get_test_results(GT_data, x, args['mode'])
            print("Model: " + str(frame_results))
            this_scale_results[args['dict_entry_name']]['Upscaling time'].append(inference_this_frame)
            for k in frame_results.keys():
                this_scale_results[args['dict_entry_name']][k].append(frame_results[k])


            if(scale_factor_in_testing not in all_results.keys()):
                all_results[scale_factor_in_testing] = {}

            for k1 in this_scale_results.keys():                
                all_results[scale_factor_in_testing][k1] = this_scale_results[k1]
    
    save_obj(all_results, results_location)
    print("Saved results")
