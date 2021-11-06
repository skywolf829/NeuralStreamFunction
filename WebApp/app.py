from flask import Flask, render_template, Response, jsonify, request, json
import os
from datetime import datetime
import sys
import base64
import cv2
from werkzeug import datastructures
from options import load_options
from datasets import LocalTemporalDataset, LocalDataset
from models import load_model
import random
from utility_functions import to_img, to_pixel_samples, PSNR
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from torchvision.utils import make_grid

file_folder_path = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(file_folder_path, 'SR_Sensitivity', 
    'templates')
static_folder = os.path.join(file_folder_path, 'SR_Sensitivity', 
    'static')
app = Flask(__name__, template_folder=template_folder, 
    static_folder=static_folder)

global SR_model
global dataset
global hr
global pixel_target
global model_name
global cropping_resolution
global continuous_scale_factor
global feature_maps

model_name = "isomag2D_RDN5_64_Shuffle4"
SR_model = None
dataset = None
hr = None
cropping_resolution = 48
continuous_scale_factor = 3.00
pixel_target = [0.0, 0.0]
feature_maps = None

def load_model_and_dataset():
    global continuous_scale_factor    
    global SR_model
    SR_model = load_current_model()
    opt = SR_model.opt
    if "TV" in opt['mode']:
        dataset = LocalTemporalDataset(opt)
    else:
        dataset = LocalDataset(opt)
    if not SR_model.upscaling_model.continuous:
        continuous_scale_factor = round(1/opt['spatial_downscale_ratio'])
    return SR_model, dataset

def load_current_model():
    global model_name
    global SR_model
    global cropping_resolution
    global continuous_scale_factor  

    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    save_folder = os.path.join(project_folder_path, "SavedModels")
    opt = load_options(os.path.join(save_folder, model_name))
    opt["device"] = "cuda:0"
    opt["save_name"] = model_name   
    opt['cropping_resolution'] = cropping_resolution
    SR_model = load_model(opt,"cuda:0")
    SR_model = SR_model.to("cuda:0")
    if not SR_model.upscaling_model.continuous:
        continuous_scale_factor = round(1/opt['spatial_downscale_ratio'])
    return SR_model

def log_visitor():
    visitor_ip = request.remote_addr
    visitor_requested_path = request.full_path
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")

    pth = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(pth,"log.txt"), "a")
    f.write(dt + ": " + str(visitor_ip) + " " + str(visitor_requested_path) + "\n")
    f.close()

@app.route('/')
def index():
    log_visitor()
    return render_template('index.html')

@app.route('/get_random_item')
def get_random_item():
    global SR_model
    global dataset
    global hr
    if SR_model is None:
        SR_model, dataset = load_model_and_dataset()

    dataset_item_no = random.randint(0, len(dataset)-1)
    hr = dataset[dataset_item_no].unsqueeze(0)
    img = to_img(hr/hr.max(), SR_model.opt['mode'], normalize=False)

    success, return_img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify({"img":str(base64.b64encode(return_img))})

@app.route('/perform_SR')
def perform_SR():
    global SR_model
    global dataset
    global hr
    global continuous_scale_factor
    if SR_model is None:
        SR_model, dataset = load_model_and_dataset()
    if hr is None:
        _ = get_random_item()
    print(hr.shape)
    hr_im = to_img(hr/hr.max(), SR_model.opt['mode'], normalize=False)
    hr = hr.to(SR_model.opt['device'])
    real_shape = hr.shape
    size = []
    sf_to_use = continuous_scale_factor if SR_model.upscaling_model.continuous \
        else (1/SR_model.opt['spatial_downscale_ratio'])
    for i in range(2, len(hr.shape)):
        size.append(round(hr.shape[i]/sf_to_use))
        
    lr = F.interpolate(hr, size=size, 
            mode='bilinear' if SR_model.opt['mode'] == "2D" else "trilinear",
            align_corners=False, recompute_scale_factor=False)
    lr_up = F.interpolate(lr, size=hr.shape[2:], mode='nearest')
    lr_im = to_img(lr_up/hr.max(), SR_model.opt['mode'], normalize=False)

    print(hr.device)
    print(lr.device)

    if(SR_model.upscaling_model.continuous):
        hr_coords, _ = to_pixel_samples(hr, flatten=False)
        cell_sizes = torch.ones_like(hr_coords, device=hr_coords.device)
        print(hr_coords.device)
        print(cell_sizes.device)
        for i in range(cell_sizes.shape[-1]):
            cell_sizes[:,:,i] *= 2 / real_shape[2+i]
        
        lr_upscaled = SR_model(lr, hr_coords, cell_sizes)
        if(SR_model.opt['mode'] == "2D"):
            lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
        else:                    
            lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
        #lr_upscaled = torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0)
    else:
        lr_upscaled = SR_model(lr)
        
    l1 = torch.abs(lr_upscaled-hr).mean().item()
    psnr = PSNR(lr_upscaled, hr).item()
    sr_im = to_img(lr_upscaled/hr.max(), SR_model.opt['mode'], normalize=False)

    success, hr_im = cv2.imencode(".png", cv2.cvtColor(hr_im, cv2.COLOR_BGR2RGB))
    hr_im = hr_im.tobytes()
    success, sr_im = cv2.imencode(".png", cv2.cvtColor(sr_im, cv2.COLOR_BGR2RGB))
    sr_im = sr_im.tobytes()
    success, lr_im = cv2.imencode(".png", cv2.cvtColor(lr_im, cv2.COLOR_BGR2RGB))
    lr_im = lr_im.tobytes()
    return jsonify(
            {
                "gt_img":str(base64.b64encode(hr_im)),
                "sr_img":str(base64.b64encode(sr_im)),
                "lr_img":str(base64.b64encode(lr_im)),
                "l1": "%0.04f" % l1,
                "psnr": "%0.02f" % psnr
            }
        )

@app.route('/sensetivity_adjustment')
def sensetivity_adjustment():
    global SR_model
    global dataset
    global hr
    global pixel_target
    global feature_maps
    if SR_model is None:
        SR_model, dataset = load_model_and_dataset()
    if hr is None:
        _ = get_random_item()
    change = float(request.args.get('change'))
    hr = hr.to(SR_model.opt['device'])
    real_shape = hr.shape
    size = []
    sf_to_use = continuous_scale_factor if SR_model.upscaling_model.continuous \
        else (1/SR_model.opt['spatial_downscale_ratio'])
    for i in range(2, len(hr.shape)):
        size.append(round(hr.shape[i]/sf_to_use))

    lr = F.interpolate(hr, size=size, 
            mode='bilinear' if SR_model.opt['mode'] == "2D" else "trilinear",
            align_corners=False, recompute_scale_factor=False)

    lr_pix = [int(pixel_target[0]*lr.shape[2]/2 + lr.shape[2]/2), 
        int(pixel_target[1]*lr.shape[3]/2 + lr.shape[3]/2)]
    lr[:,:,lr_pix[0],lr_pix[1]] += change

    lr_up = F.interpolate(lr, size=hr.shape[2:], mode='nearest')
    lr_im = to_img(lr_up/hr.max(), SR_model.opt['mode'], normalize=False)

    lr.requires_grad = True
    if(SR_model.upscaling_model.continuous):
        hr_coords, _ = to_pixel_samples(hr, flatten=False)
        cell_sizes = torch.ones_like(hr_coords)

        for i in range(cell_sizes.shape[-1]):
            cell_sizes[:,:,i] *= 2 / real_shape[2+i]
        
        feature_maps = SR_model.feature_extractor(lr.clone().detach()).clone().detach()
        lr_upscaled = SR_model(lr, hr_coords, cell_sizes)
        if(SR_model.opt['mode'] == "2D"):
            lr_upscaled = lr_upscaled.permute(2, 0, 1).unsqueeze(0)
        else:                    
            lr_upscaled = lr_upscaled.permute(3, 0, 1, 2).unsqueeze(0)
        #lr_upscaled = torch.flatten(lr_upscaled,start_dim=1, end_dim=-1).permute(1,0)
    else:
        feature_maps = SR_model.feature_extractor(lr.clone().detach()).clone().detach()
        lr_upscaled = SR_model(lr)
    
    changed_sr_im = to_img(lr_upscaled/hr.max(), SR_model.opt['mode'], normalize=False)
    imageio.imwrite("test.png", changed_sr_im)
    sense = torch.autograd.grad(outputs = [lr_upscaled], 
        inputs = [lr], grad_outputs = torch.ones_like(lr_upscaled) ,
        allow_unused=True, retain_graph=True, create_graph=True)[0]
    
    sense = torch.abs(sense)
    sense *= (1/sense.max())
    sense = F.interpolate(sense, size=hr.shape[2:], mode='nearest')
    sense_img = sense[0,0]*255
    sense_img = sense_img.detach().cpu().numpy().astype(np.uint8)
    

    success, changed_sr_im = cv2.imencode(".png", cv2.cvtColor(changed_sr_im, cv2.COLOR_BGR2RGB))
    changed_sr_im = changed_sr_im.tobytes()
    success, changed_lr_img = cv2.imencode(".png", cv2.cvtColor(lr_im, cv2.COLOR_BGR2RGB))
    changed_lr_img = changed_lr_img.tobytes()
    success, sense_img = cv2.imencode(".png", cv2.cvtColor(sense_img, cv2.COLOR_BGR2RGB))
    sense_img = sense_img.tobytes()

    return jsonify(
            {
                "changed_sr_img":str(base64.b64encode(changed_sr_im)),
                "changed_lr_img":str(base64.b64encode(changed_lr_img)),
                "sensitivity_img":str(base64.b64encode(sense_img))
            }
        )

@app.route('/update_feature_maps')
def update_feature_maps():
    global feature_maps
    fm = feature_maps.permute(1, 0, 2, 3)
    img = 255*make_grid(fm, nrow=10, normalize=True).permute(1, 2, 0).cpu().numpy()
    success, return_img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify(
        {
            "img":str(base64.b64encode(return_img)),
            "min": "%0.03f" % feature_maps.min().cpu().item(),            
            "max": "%0.03f" % feature_maps.max().cpu().item()
        }
    )

@app.route('/get_available_models')
def get_available_models():
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    model_names = os.listdir(save_folder)
    return jsonify(
            { "model_names": model_names }
    )

@app.route('/load_new_model')
def load_new_model():
    global model_name
    global SR_model
    model_name = str(request.args.get('new_model_name'))
    SR_model = load_current_model()
    return jsonify({"success": True})
    
@app.route('/change_crop')
def change_crop():
    global cropping_resolution
    global SR_model
    global dataset
    cropping_resolution = int(request.args.get('cropping_resolution'))
    SR_model.opt['cropping_resolution'] = cropping_resolution
    dataset.opt['cropping_resolution'] = cropping_resolution
    return jsonify({"success": True})
    
@app.route('/change_scale_factor')
def change_scale_factor():
    global SR_model
    global dataset
    global continuous_scale_factor
    continuous_scale_factor = round(float(request.args.get('scale_factor')))
    return jsonify({"success": True})
    


if __name__ == '__main__':
    
    SR_model, dataset = load_model_and_dataset()
    
    app.run(host='127.0.0.1',debug=True,port="12345")
    #app.run(host='0.0.0.0',debug=False,port="80")