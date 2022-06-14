# Neural Dual Stream Function
This project uses neural networks to model vector fields for visualization.


## Installation

We recommend conda for Python package management. To install, run the following:
```
conda env create --file env.yml
conda activate NDSF
```

Creating the environment will take a while. It took our windows machine roughly 1 hour to create the environment.
Next, navigate to https://pytorch.org/get-started/locally/ and follow instructions to install pytorch.

For instance:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

was the command we used to install pytorch on our Windows machine for testing.

This code has been tested on:

Windows 10 with Python 3.9.12, Pytorch 1.11 (with cuda 11.3)

Ubuntu 20.04 LTS with Python 3.9.12, Pytorch 1.11 (with cuda 11.3)

MacOS Monterey 12.4 with Python 3.9.12, Pytorch 1.13 (nightly build with MPS on Apple M1)


## Examples

Below we list examples for our code

