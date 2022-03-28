import os
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import argparse
import h5py
import time

def PSNR(x, y, range = 1.0):
    return 20*np.log10(range) - \
        10*np.log10(((y-x)**2).mean())
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--file',default="4010.h5",type=str,help='File to test compression on')
    parser.add_argument('--dims',default=2,type=int,help='# dimensions')    
    parser.add_argument('--channels',default=1,type=int,help='# channels')
    parser.add_argument('--nx',default=1024,type=int,help='# x dimension')
    parser.add_argument('--ny',default=1024,type=int,help='# y dimension')
    parser.add_argument('--nz',default=1024,type=int,help='# z dimension')
    parser.add_argument('--value',default=40,type=float,help='Target metric')
    parser.add_argument('--metric',default='psnr',type=str)
    parser.add_argument('--save_netcdf',default="false",type=str2bool)
    parser.add_argument('--device',default="cpu",type=str)
    

    args = vars(parser.parse_args())

    folder_path = os.path.dirname(os.path.abspath(__file__))
    
    f = h5py.File(os.path.join(folder_path, args['file']), "r")
    d = np.array(f['data'], dtype=np.float32)
    f.close()
    for i in range(args['channels']):
        d[i].tofile(args['file'] + ".dat")

    value = args['value']
    data_channels = []
    f_size_kb = 0
    for i in range(args['channels']):            
        print(f"Channel {i}")
        command = "./tthresh -i " + args['file'] + ".dat -s " + \
            str(args['nx']) + " " + str(args['ny'])
        if(args['dims'] == 3):
            command = command + " " + str(args['nz'])
        if(args['metric'] == "psnr"):
            command = command + " -p " + str(value)
        elif(args['metric'] == "mre"):
            command = command + " -e " + str(value)
        command = command + " -t float -c " + args['file'] + ".dat.tthresh"
        start_t = time.time()
        print("Running: " + command)
        os.system(command)
        compression_time = time.time() - start_t

        f_size_kb += os.path.getsize(args['file'] + ".dat.tthresh") / 1024

        command = "./tthresh -c " + args['file'] + ".dat.tthresh -o " + args['file'] + ".dat.tthresh.out"  

        os.system(command)

        dc = np.fromfile(args['file']+".dat.tthresh.out", dtype=np.float32)
        if(args['dims'] == 2):
            dc = dc.reshape(args['ny'], args['nx'])
        elif(args['dims'] == 3):
            dc = dc.reshape(args['nz'], args['ny'], args['nx'])    
        data_channels.append(dc)
        command = "mv " + args['file']+".dat.tthresh " + folder_path + \
            "/psnr_"+str(value)+"_"+args['file']+"_"+str(i)+".tthresh"
        os.system(command)
    dc = np.stack(data_channels)

    rec_psnr = PSNR(dc, d)
    
    print("Target: " +args['metric'] + " " + str(value))
    print("Actual: " +str(rec_psnr))

    if(args['save_netcdf']):
        from netCDF4 import Dataset
        rootgrp = Dataset("./tthresh_"+args['file']+".nc", "w", format="NETCDF4")
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

    

    os.remove(args['file']+'.dat')    
    os.remove(args['file']+'.dat.tthresh.out')