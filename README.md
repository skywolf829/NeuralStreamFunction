# Neural Dual Stream Function
This project uses neural networks to model vector fields for visualization.


## Installation

We recommend conda for Python package management. To install, run the following:
```
conda env create --file env.yml
conda activate NeuralStreamFunction
```

Creating the environment will take a while. If the above fails, use this as a backup:
```
conda create --name NeuralStreamFunction python=3.9
conda activate NeuralStreamFunction
conda install netcdf4 vtk zeep opencv flask imageio h5py scikit-image pandas matplotlib tensorboard --channel conda-forge
```
Both approaches should do the same thing and leave you with the same environment.

Once thats finished and the environment has been activated, navigate to https://pytorch.org/get-started/locally/ and follow instructions to install pytorch on your machine.

For instance:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

was the command we used to install pytorch on our Windows machine for testing.

This code has been tested on:

Windows 10 with Python 3.9.12, Pytorch 1.13.1 (with CUDA 11.6)

Ubuntu 20.04.4 LTS with Python 3.9.13, Pytorch 1.12.1 (with CUDA 11.4)

MacOS Ventura 13.1 with Python 3.9.15, Pytorch 1.13.1 (with CPU on Apple M1 - MPS is not supported fully, but performance will improve when it is). 

To download our data (hosted on Google Drive - 357MB compressed/774MB decompressed), we recommend using the conda package gdown.
```
conda install gdown -c conda-forge
gdown 10EUDp7D7B6-LokxEHrfcoggixQ5jL998
tar -xvf data.tar.gz
rm data.tar.gz
```
Now, all our test data should be in Data/ as NetCDF files, which can readily be visualized in ParaView.

## Usage

### ```start_jobs.py```
This script is responsible for starting a set of jobs hosted in a JSON file in /Code/Batch_run_settings, and issuing each job to available GPUs on the system. The jobs in the JSON file can be training (```train.py```) or testing (```test.py```), and one job will be addressed to each device available for training/testing. When a job completes on a device, the device is released and becomes available for other jobs to be designated that device. The jobs are not run in sequential order unless you only have 1 device, so do not expect this script to train+test a model sequentially unless you use only one device.

Command line arguments are:
```--settings```: the .json file (located in /Code/Batch_run_settings/) with the training/testing setups to run. See the examples in the folder for how to create a new settings file. Required argument.

```--devices```: the list of devices (comma separated) to issue jobs to. By default, "all" available CUDA devices are used. If no CUDA devices are detected, the CPU is used. 

```--data_devices```: the device to host the data (vector field) on. In some cases, the data may be too large to host on the same GPU that training is happening on, using system RAM instead of GPU VRAM may be preferred. Options are "same", implying using the same device for the data as is used for the model, and "cpu", which puts the data on system RAM. Default is "same".

#### Example usage:

The following will run the jobs defined in example_file.json on all available CUDA devices (if available) or the CPU if no CUDA devices are detected by PyTorch. The vector field data will be hosted on the same device that the models train on.

```python Code/start_jobs.py --settings example_file.json```

The following will run the jobs defined in example_file.json on cuda devices 1, 2, 4, and 7, with the data hosted on the same device.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cuda:2,cuda:4,cuda:7```

The following will run the jobs defined in example_file.json on cuda devices 1 and 2, with the data hosted on the system memory.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cuda:2 --data_devices cpu```

The following will run the jobs defined in example_file.json on cuda devices 1 and cpu, with the data hosted on the same devices as the model.

```python Code/start_jobs.py --settings example_file.json --devices cuda:1,cpu```

(For M1 Macs with MPS) - The following will run the jobs defined in example_file.json on MPS (metal performance shaders), which is Apple's hardware for acceleration. Many PyTorch functions are not yet implemented for MPS as of Torch version 1.13.1, and as such our code cannot natively run on MPS at this time, but as more released of PyTorch come out, we expect this to run without issue in the future.

```python Code/start_jobs.py --settings example_file.json --devices mps```