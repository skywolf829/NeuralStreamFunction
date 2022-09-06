# Neural Stream Function
This project uses neural networks to model vector fields for visualization.


## Installation

We recommend conda for Python package management. To install, run the following:
```
conda env create --file env.yml
conda activate NDSF
```

Creating the environment will take a while. If the above fails, use this as a backup:
```
conda create --name NDSF python=3.9
conda activate NDSF
conda install netcdf4 vtk zeep opencv flask imageio h5py scikit-image pandas numba matplotlib tensorboard --channel conda-forge
```
Both approaches should do the same thing and leave you with the same environment.

Once thats finished and the environment has been activated, navigate to https://pytorch.org/get-started/locally/ and follow instructions to install pytorch on your machine.

For instance:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

was the command we used to install pytorch on our Windows machine for testing.

This code has been tested on:

Windows 10 with Python 3.9.12, Pytorch 1.11 (with cuda 11.3)

Ubuntu 20.04 LTS with Python 3.9.12, Pytorch 1.11 (with cuda 11.3)

MacOS Monterey 12.4 with Python 3.9.12, Pytorch 1.13 (nightly build with MPS on Apple M1)

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

