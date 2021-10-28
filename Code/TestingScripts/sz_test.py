import os
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import imageio
import argparse
import h5py
import time
import torch
from utility_functions import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--file',default="4010.h5",type=str,help='File to test compression on')
    parser.add_argument('--dims',default=2,type=int,help='# dimensions')    
    parser.add_argument('--channels',default=1,type=int,help='# channels')
    parser.add_argument('--nx',default=1024,type=int,help='# x dimension')
    parser.add_argument('--ny',default=1024,type=int,help='# y dimension')
    parser.add_argument('--nz',default=1024,type=int,help='# z dimension')
    parser.add_argument('--output_folder',default="mag2D_4010",type=str,help='Where to save results')
    parser.add_argument('--start_value',default=10,type=float,help='PSNR to start tests at')
    parser.add_argument('--end_value',default=100,type=float,help='PSNR to end tests at')
    parser.add_argument('--value_skip',default=10,type=float,help='PSNR increment by')
    parser.add_argument('--metric',default='psnr',type=str)
    parser.add_argument('--save_netcdf',default="false",type=str2bool)
    parser.add_argument('--device',default="cpu",type=str)
    

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data", "DataReduction")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(output_folder, args['output_folder'])

    if(not os.path.exists(save_folder)):
        os.makedirs(save_folder)
    
    # psnr -> metrics
    results = {}
    results['file_size'] = []
    results['psnrs'] = []
    results['rec_psnr'] = []
    results['rec_ssim'] = []
    results['compression_time'] = []
    results['rec_mre'] = []
    results['rec_pwmre'] = []
    results['rec_inner_mre'] = []
    results['rec_inner_pwmre'] = []
    
    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
        if("SZ" in all_data.keys()):
            results = all_data['SZ']
        else:
            all_data['SZ'] = results
    else:
        all_data = {}
        all_data['SZ'] = results

    f = h5py.File(os.path.join(data_folder, args['file']), "r")
    d = np.array(f['data'])
    f.close()
    for i in range(args['channels']):
        d[i].tofile(args['file'] + ".dat")

    value = args['start_value']
    while(value < args['end_value']):
        data_channels = []
        f_size_kb = 0
        for i in range(args['channels']):            
            command = "sz -z -f -i " + args['file'] + ".dat -" + str(args['dims']) + " " + \
                str(args['nx']) + " " + str(args['ny'])
            if(args['dims'] == 3):
                command = command + " " + str(args['nz'])
            if(args['metric'] == "psnr"):
                command = command + " -M PSNR -S " + str(value)
            elif(args['metric'] == "mre"):
                command = command + " -M REL -R " + str(value)
            elif(args['metric'] == "pw_mre"):
                command = command + " -M PW_REL -P " + str(value)
            start_t = time.time()
            print("Running: " + command)
            os.system(command)
            compression_time = time.time() - start_t

            f_size_kb += os.path.getsize(args['file'] + ".dat.sz") / 1024

            command = "sz -x -f -s " + args['file'] + ".dat.sz -" + str(args['dims']) + " " + \
                str(args['nx']) + " " + str(args['ny'])
            if(args['dims'] == 3):
                command = command + " " + str(args['nz'])
            command = command + " -S " + str(value) 

            os.system(command)

            dc = np.fromfile(args['file']+".dat.sz.out")
            dc.dtype = np.float32
            if(args['dims'] == 2):
                dc = dc.reshape(args['nx'], args['ny'])
            elif(args['dims'] == 3):
                dc = dc.reshape(args['nx'], args['ny'], args['nz'])    
            data_channels.append(dc)
            command = "mv " + args['file']+".dat.sz " + save_folder + \
                "/psnr_"+str(value)+"_"+args['file']+"_"+str(i)+".sz"
            os.system(command)
        dc = np.stack(data_channels)

        rec_psnr = PSNR(dc, d)
        final_mre : float = relative_error(dc, d).item()
        final_pwmre: float = pw_relative_error(dc, d).item()
        #rec_ssim = ssim(d, dc)
                
        if(args['dims'] == 2):
            rec_ssim = ssim(torch.Tensor(dc).unsqueeze(0).to(args['device']), 
                torch.Tensor(d).unsqueeze(0).to(args['device'])).cpu().item()
            inner_mre = relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20])
            inner_pwmre = pw_relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20])
        elif(args['dims'] == 3):      
            rec_ssim = ssim3D(torch.Tensor(dc).unsqueeze(0), torch.Tensor(d).unsqueeze(0)).cpu().item()
            inner_mre = relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20,20:dc.shape[3]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20,20:d.shape[3]-20])
            inner_pwmre = pw_relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20,20:dc.shape[3]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20,20:d.shape[3]-20])
        im = toImg(dc, "2D" if args['dims'] == 2 else "3D")
        imageio.imwrite(os.path.join(save_folder, "sz_"+args['file']+"_"+str(value)+".png"), im)

        print("Target: " +args['metric'] + " " + str(value))
        print("PSNR: " + str(rec_psnr) + " SSIM: " + str(rec_ssim))

        if(args['save_netcdf']):
            from netCDF4 import Dataset
            rootgrp = Dataset("sz_"+args['file']+".nc", "w", format="NETCDF4")
            rootgrp.createDimension("u")
            rootgrp.createDimension("v")
            if(args['dims'] == 3):
                rootgrp.createDimension("w")
            rootgrp.createDimension("channels", dc.shape[0])
            if(args['dims'] == 3):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v","w"))
            elif(args['dims'] == 2):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v"))
            dim_0[:] = dc[0]

        results['psnrs'].append(value)
        results['file_size'].append(f_size_kb)
        results['compression_time'].append(compression_time)
        results['rec_psnr'].append(rec_psnr)
        results['rec_ssim'].append(rec_ssim)
        results['rec_mre'].append(final_mre)
        results['rec_pwmre'].append(final_pwmre)
        results['rec_inner_mre'].append(inner_mre)
        results['rec_inner_pwmre'].append(inner_pwmre)
        all_data['SZ'] = results
        save_obj(all_data, os.path.join(save_folder, "results.pkl"))
        value += args['value_skip']
    

    os.remove(args['file']+'.dat')    
    os.remove(args['file']+'.dat.sz.out')