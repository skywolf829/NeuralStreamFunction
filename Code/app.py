from flask import Flask, render_template, Response, jsonify, request, json
import os
from datetime import datetime
import sys
import base64
import cv2
from werkzeug import datastructures
from options import load_options
from datasets import Dataset
from models import load_model
import random
from utility_functions import PSNR
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from torchvision.utils import make_grid


class AppModelAndController():
    def __init__(self):
        self.model = None
        self.dataset = None
        self.opt = None
        self.gt_im = None

        self.device = "cuda:0"
        self.model_name = "cat_8x512"

        self.file_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.project_folder_path = os.path.join(file_folder_path, "..")
        self.save_folder = os.path.join(self.project_folder_path, "SavedModels")

        self.load_model_from_name(self.model_name, self.device)
        self.load_dataset_from_opt(self.opt)

    def load_model_from_name(self, model_name, device="cuda:0"):
        self.gt_im = None
        self.model_name = model_name
        
        self.opt = load_options(os.path.join(self.save_folder, model_name))
        self.device = device
        self.opt["device"] = self.device
        self.opt['data_device'] = self.device
        self.opt["save_name"] = model_name   
        self.model = load_model(self.opt,self.device)
        self.model = self.model.to(self.device)
        self.load_dataset_from_opt(self.opt)
    
    def load_dataset_from_opt(self, opt):
        self.dataset = Dataset(opt)

    def get_gt(self, factor):

        gt_im = self.dataset.get_2D_slice()
        gt_im -= self.dataset.min()
        gt_im /= (self.dataset.max() - self.dataset.min())
        
        gt_im = F.interpolate(gt_im.unsqueeze(0), 
            scale_factor=factor, mode='bilinear', 
            align_corners=False)[0]

        gt_im *= 255
        gt_im = gt_im.type(torch.uint8)


        gt_im = gt_im.permute(1, 2, 0)
        gt_im = gt_im.cpu().numpy()

        return gt_im
    
    def get_gt_crop(self, starts, widths, ss_factor):

        samples = []
        for i in range(len(widths)):
            samples.append(int(ss_factor*widths[i]*self.dataset.data.shape[2+i]))    

        if(len(self.dataset.data.shape) == 5):
            starts.append(0.5)
            widths.append(1e-6)
            samples.append(1)

        gt_im = self.dataset.sample_rect(starts, widths, samples)
        gt_im = gt_im[0]

        if(len(self.dataset.data.shape) == 5):
            gt_im = gt_im[:,:,:,0]
        #print(gt_im.shape)
        
        gt_im -= self.dataset.min()
        gt_im /= (self.dataset.max() - self.dataset.min())

        gt_im *= 255
        gt_im = gt_im.type(torch.uint8).clamp(0, 255).permute(2, 1, 0).cpu().numpy()

        return gt_im
    
    def get_full_reconstruction(self, ss_factor, boundary_scaling):

        grid = list(self.dataset.data.shape[2:])
        for i in range(len(grid)):
            grid[i] *= ss_factor
            grid[i] = int(grid[i])

        with torch.no_grad():
            im = self.model.sample_grid_for_image(grid, 
                boundary_scaling=boundary_scaling)
        #print(im.min())
        #print(im.max())
        #print("dataset")
        #print(self.dataset.min())
        #print(self.dataset.max())
        im -= self.dataset.min()
        im /= (self.dataset.max()-self.dataset.min())
        im *= 255
        im = im.clamp(0, 255)
        im = im.type(torch.uint8).permute(1, 0, 2)

        return im.cpu().numpy()

    def get_crop(self, starts, widths, ss_factor, boundary_scaling):    
        samples = []
        for i in range(len(widths)):
            samples.append(int(ss_factor*widths[i]*self.dataset.data.shape[2+i]))  
            starts[i] -= 0.5
            starts[i] *= (2*boundary_scaling)
            widths[i] *= (2*boundary_scaling)   
        if(len(self.dataset.data.shape) == 5):
            starts.append(0.0)
            widths.append(1e-6)
            samples.append(1)

        with torch.no_grad():
            im = self.model.sample_rect(starts, widths, samples)

        if(len(self.dataset.data.shape) == 5):
            im = im[:,:,0,:]

        im -= self.dataset.min()
        im /= (self.dataset.max()-self.dataset.min())
        
        im *= 255
        im = im.clamp(0, 255)
        im = im.type(torch.uint8).permute(1, 0, 2)

        return im.cpu().numpy()

    def get_grad(self, ss_factor, boundary_scaling, input_dim, output_dim):
        grid = list(self.dataset.data.shape[2:])
        for i in range(len(grid)):
            grid[i] *= ss_factor
            grid[i] = int(grid[i])

        im = self.model.sample_grad_grid_for_image(grid, 
            boundary_scaling=boundary_scaling, 
            input_dim = input_dim,
            output_dim=output_dim)

        im -= im.min()
        im /= im.max()
        im *= 255
        im = im.clamp(0, 255)
        im = im.type(torch.uint8).permute(1, 0, 2)

        return im.cpu().numpy()

    def get_grad_crop(self, starts, widths, ss_factor, boundary_scaling,
        input_dim, output_dim):    
        samples = []
        for i in range(len(widths)):
            samples.append(int(ss_factor*widths[i]*self.dataset.data.shape[2+i]))  
            starts[i] -= 0.5
            starts[i] *= (2*boundary_scaling)
            widths[i] *= (2*boundary_scaling)   
        if(len(self.dataset.data.shape) == 5):
            starts.append(0.0)
            widths.append(1e-6)
            samples.append(1)

        with torch.no_grad():
            im = self.model.sample_grad_rect(starts, widths, samples, 
                input_dim, output_dim)

        if(len(self.dataset.data.shape) == 5):
            im = im[:,:,0,:]

        im -= im.min()
        im /= im.max()
        
        im *= 255
        im = im.clamp(0, 255)
        im = im.type(torch.uint8).permute(1, 0, 2)

        return im.cpu().numpy()


file_folder_path = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(file_folder_path, "..", 'App', 
    'templates')
static_folder = os.path.join(file_folder_path, "..", 'App', 
    'static')
app = Flask(__name__, template_folder=template_folder, 
    static_folder=static_folder)

global amc
amc = AppModelAndController()

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

@app.route('/get_gt')
def get_gt():
    global amc
    
    factor = float(request.args.get('scale_factor'))
    im = amc.get_gt(factor)


    print("GT")
    print(im.shape)

    success, return_img = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify(
        {
            "img":str(base64.b64encode(return_img))
        }
    )

@app.route('/get_full_reconstruction')
def get_full_reconstruction():
    global amc
    factor = float(request.args.get('scale_factor'))
    boundary_scaling = float(request.args.get('boundary_scaling'))
    
    im = amc.get_full_reconstruction(factor, boundary_scaling)

    print("Recon")
    print(im.shape)

    success, return_img = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify(
        {
            "img":str(base64.b64encode(return_img))
        }
    )

@app.route('/get_full_gradient')
def get_full_gradient():
    global amc
    factor = float(request.args.get('scale_factor'))
    boundary_scaling = float(request.args.get('boundary_scaling'))
    input_dim = int(request.args.get('input_dim'))
    output_dim = int(request.args.get('output_dim'))
    
    im = amc.get_grad(factor, boundary_scaling, input_dim, output_dim)

    print("Full grad")
    print(im.shape)

    success, return_img = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify(
        {
            "img":str(base64.b64encode(return_img))
        }
    )

@app.route('/get_crop')
def get_crop():
    global amc
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    width = float(request.args.get('width'))
    height = float(request.args.get('height'))
    factor = float(request.args.get('scale_factor'))
    boundary_scaling = float(request.args.get('boundary_scaling'))

    im = amc.get_crop([x, y], [width, height], factor, boundary_scaling)
    print("crop")
    print(im.shape)

    success, return_img = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify(
        {
            "img":str(base64.b64encode(return_img))
        }
    )

@app.route('/get_grad_crop')
def get_grad_crop():
    global amc
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    width = float(request.args.get('width'))
    height = float(request.args.get('height'))
    factor = float(request.args.get('scale_factor'))
    boundary_scaling = float(request.args.get('boundary_scaling'))
    input_dim = float(request.args.get('input_dim'))
    output_dim = float(request.args.get('output_dim'))

    im = amc.get_grad_crop([x, y], [width, height], factor, boundary_scaling,
        input_dim, output_dim)
    print("grad crop")
    print(im.shape)

    success, return_img = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify(
        {
            "img":str(base64.b64encode(return_img))
        }
    )

@app.route('/get_gt_crop')
def get_gt_crop():
    global amc
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    width = float(request.args.get('width'))
    height = float(request.args.get('height'))
    factor = float(request.args.get('scale_factor'))

    im = amc.get_gt_crop([x, y], [width, height], factor)
    print("gt crop")
    print(im.shape)

    success, return_img = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return_img = return_img.tobytes()
    return jsonify(
        {
            "img":str(base64.b64encode(return_img))
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
    global amc
    amc.load_model_from_name(str(request.args.get('new_model_name')))
    return jsonify({"success": True})
    

if __name__ == '__main__':    
    app.run(host='127.0.0.1',debug=True,port="12345")
    #app.run(host='0.0.0.0',debug=False,port="80")