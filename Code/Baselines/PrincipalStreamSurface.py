from __future__ import absolute_import, division, print_function
import argparse
from Datasets.datasets import Dataset
import datetime
from Other.utility_functions import normal
import torch
import time
import os
from Models.options import *

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def princpal_stream_function(vf):
    sf = torch.zeros([1, 1, vf.shape[2], vf.shape[3], vf.shape[4]],
                     device = vf.device)
    
    return sf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device',default=None, type=str,
        help='Which device to use')
    parser.add_argument('--data_device',default=None,type=str,
        help='Which device to keep the data on')
    parser.add_argument('--data',default=None,type=str,
        help='Data file name')

    args = vars(parser.parse_args())

    torch.manual_seed(11235813)
    
    opt = Options.get_default()

    for k in args.keys():
        if args[k] is not None:
            opt[k] = args[k]
        dataset = Dataset(opt)
        
    now = datetime.datetime.now()
    start_time = time.time()    
    
    
    sf = princpal_stream_function(dataset.data)