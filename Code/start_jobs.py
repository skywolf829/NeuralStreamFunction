import torch
import os
import argparse
import json
import time
import subprocess
import shlex
from Other.utility_functions import create_path

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def build_commands(settings_path):
    f = open(settings_path)
    data = json.load(f)
    commands = []
    command_names = []
    log_locations = []
    for run_name in data.keys():
        command_names.append(run_name)
        script_name = data[run_name][0]
        variables = data[run_name][1]
        if(script_name == "train.py"):
            command = "python Code/" + str(script_name) + " "
        elif(script_name == "test_model.py"):
            command = "python Code/Tests/" + str(script_name) + " "
            
        for var_name in variables.keys():
            command = command + "--" + str(var_name) + " "
            command = command + str(variables[var_name]) + " "
        commands.append(command)
        if(script_name == "train.py"):
            log_locations.append(os.path.join(save_folder, variables["save_name"], "train_log.txt"))
        elif(script_name == "test_model.py"):
            log_locations.append(os.path.join(save_folder,  variables['load_from'], "test_log.txt"))
    f.close()
    return command_names, commands, log_locations

def parse_devices(devices_text):
    devices = devices_text.split(',')
    for i in range(len(devices)):
        devices[i] = devices[i].strip()
        if(devices[i].isnumeric()):
            devices[i] = "cuda:"+str(devices[i])
        else:
            devices[i] = str(devices[i])
    return devices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains/tests models given settings on available gpus')
    parser.add_argument('--settings',default=None,type=str,
        help='The settings file with options for each model to train/test')
    parser.add_argument('--devices',default="all",type=str,
        help='Which [cuda] devices(s) to train on, separated with commas. Default: all, which uses all available CUDA devices')
    parser.add_argument('--data_devices',default="same",type=str,
        help='Which devices to put the training data on. "same" as model, or "cpu".')
    
    args = vars(parser.parse_args())

    settings_path = os.path.join(project_folder_path, "Code", "Batch_run_settings", args['settings'])
    command_names, commands, log_locations = build_commands(settings_path)

    if(args['devices'] == "all"):
        available_devices = []
        if(torch.cuda.is_available()):
            for i in range(torch.cuda.device_count()):
                available_devices.append("cuda:" + str(i))
        else:
            available_devices.append("cpu")
            
    else:
        available_devices = parse_devices(args['devices'])
    
    jobs_training = []
    while(len(commands) + len(jobs_training) > 0):
        # Check if any jobs have finished and a GPU is freed
        i = 0 
        while i < len(jobs_training):
            c_name, job, gpu, job_start_time = jobs_training[i]
            job_code = job.poll()
            if(job_code is not None):
                # Job has finished executing
                jobs_training.pop(i)
                job_end_time = time.time()
                print(f"Job {c_name} has finished with exit code {job_code} after {(job_end_time-job_start_time)/60 : 0.02f} minutes, freeing {gpu}")
                # The gpu is freed, added back to available_devices
                available_devices.append(gpu)
            else:
                i += 1

        # Check if any gpus are available for commands in queue
        if(len(available_devices) > 0 and len(commands)>0):
            c = commands.pop(0)
            c_name = command_names.pop(0)
            log_location = log_locations.pop(0)
            g = str(available_devices.pop(0))
            if(args['data_devices'] == "same"):
                data_device = str(g)
            else:
                data_device = "cpu"
            c = c + "--device " + g + " --data_device " + data_device
            c_split = shlex.split(c)
            # Logging location
            create_path(log_location[:-7])
            output_path = open(log_location,'a+')
            # Start the job
            print(f"Starting job {c_name} on device {g}")
            job = subprocess.Popen(c_split, stdout=output_path, stderr=output_path)
            jobs_training.append((c_name, job, g, time.time()))
        else:
            # Otherwise wait
            time.sleep(1.0)

    print("All jobs have completed.")
    quit()