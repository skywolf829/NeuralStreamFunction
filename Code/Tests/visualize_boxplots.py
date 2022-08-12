from __future__ import absolute_import, division, print_function
import argparse
import os
from Other.utility_functions import jacobian, curl, nc_to_tensor, tensor_to_cdf
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--folder',default=None,type=str,help="Model name to load")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    folder_with_npys = os.path.join(output_folder, args['folder'])

    arrays = []
    names = []
    for filename in os.listdir(folder_with_npys):
        a = np.load(os.path.join(folder_with_npys, filename))
        arrays.append(a)
        names.append(filename.split('.')[0])

    plt.boxplot(arrays, vert=False, showfliers=False, labels=names)
    plt.show()
    
    
        
    
        



        

