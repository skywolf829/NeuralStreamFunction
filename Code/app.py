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
        self.model_name = model_name
        
        self.opt = load_options(os.path.join(self.save_folder, model_name))
        self.device = device
        self.opt["device"] = self.device
        self.opt['data_device'] = self.device
        self.opt["save_name"] = model_name   
        self.model = load_model(self.opt,self.device)
        self.model = self.model.to(self.device)
    
    def load_dataset_from_opt(self, opt):
        self.dataset = Dataset(opt)

    def get_gt(self):
        if(self.gt_im is None):
            self.gt_im = self.dataset.get_2D_slice().clone()
            im_min = self.dataset.min()
            self.gt_im -= im_min
            im_max = self.dataset.max()
            self.gt_im /= im_max
            
            self.gt_im *= 255
            self.gt_im = self.gt_im.type(torch.uint8)

            self.gt_im = self.gt_im.permute(1, 2, 0)
            self.gt_im = self.gt_im.cpu().numpy()
        return self.gt_im
    
    def get_full_reconstruction(self):
        grid_to_sample = self.dataset.data.shape[2:]
        with torch.no_grad():
            im = self.model.sample_grid(grid_to_sample)
        if(self.dataset.min() < 0):
            im -= self.dataset.min()
            im /= self.dataset.max()
        
        im *= 255
        im = im.type(torch.uint8)

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
    
    im = amc.get_gt()

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
    
    im = amc.get_full_reconstruction()

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