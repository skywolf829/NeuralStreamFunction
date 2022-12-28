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

Windows 10 with Python 3.9.12, Pytorch 1.13.1 (with cuda 11.6)

Ubuntu 20.04 LTS with Python 3.9.12, Pytorch 1.13.1 (with cuda 11.6)

MacOS Ventura 13.1 with Python 3.9.12, Pytorch 1.13.1 (with both CPU and MPS on Apple M1)

To download our data (hosted on Google Drive), we recommend using the python package gdown.
```
conda install gdown --channel conda-forge
gdown 1V8eypZZ9lW2082QTVfwf7w51eq0eHDTY
tar -xvf data.tar.gz
rm data.tar.gz
```
Now, all our test data should be in Data/ as NetCDF files, which can readily be visualized in ParaView.

## Examples

Below we list examples for our code.

