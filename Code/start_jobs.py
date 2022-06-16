import torch
import os
import argparse
import json
import time
import subprocess
import shlex

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
        command = "python Code/train.py "
        for var_name in data[run_name].keys():
            command = command + "--" + str(var_name) + " "
            command = command + str(data[run_name][var_name]) + " "
        commands.append(command)        
        log_locations.append(os.path.join(save_folder, data[run_name]["save_name"]), "log.txt")
    f.close()
    return command_names, commands, log_locations

def parse_gpus(gpus_text):
    gpus = gpus_text.split(',')
    for i in range(len(gpus)):
        gpus[i] = gpus[i].strip()
        gpus[i] = "cuda:"+str(gpus[i])
    return gpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains models given settings on available gpus')
    parser.add_argument('--settings_file',default=None,type=str,
        help='The settings file with options for each model to train')
    parser.add_argument('--gpus',default="all",type=str,
        help='Which [cuda] GPU(s) to train on, separated with commas. Default: all')
    args = vars(parser.parse_args())

    settings_path = os.path.join(project_folder_path, "Code", "TrainingSettings", args['settings_file'])
    command_names, commands, log_locations = build_commands(settings_path)

    if(args['gpus'] == "all"):
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    else:
        available_gpus = parse_gpus(args['gpus'])
    
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
                print(f"Job {c_name} has finished with exit code {job_code} after {(job_end_time-job_start_time)/60} minutes")
                # The gpu is freed, added back to available_gpus
                available_gpus.append(gpu)
            else:
                i += 1

        # Check if any gpus are available for commands in queue
        if(len(available_gpus) > 0 and len(commands)>0):
            c = commands.pop(0)
            c_name = command_names.pop(0)
            log_location = log_locations.pop(0)
            g = available_gpus.pop(0)
            c = c + "--device " + str(g) + " --data_device " + str(g)
            c_split = shlex.split(c)
            # Logging location
            output_path = open(log_location,'w+')
            # Start the job
            print(f"Starting job {c_name} on device {g}")
            job = subprocess.Popen(c_split, stdout=output_path, stderr=output_path)
            jobs_training.append((c_name, job, g, time.time()))
        else:
            # Otherwise wait
            time.sleep(1.0)

    print("All training has completed.")
    quit()